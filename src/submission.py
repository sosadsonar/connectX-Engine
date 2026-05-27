%%writefile submission.py
import os
import subprocess
import ctypes
import sys

CHAMPION_WEIGHTS = {
    "WIN_BASE": 10000000,
    "FORK_SCORE": 613562,
    "THREAT_SCORE": 33787,
    "MAX_STRATEGIC": 13194,
    "C_SMOOTH": 1625.6527982367497,
    "K_EDGE": 23.908668315969653,
    "K_CORNER": 9.505983248984466,
    "CNN_POWER": 1.998149965005146,
    "ALPHA_BALANCED": 0.7123610249973652,
    "ALPHA_DEFENSIVE": 1.4855626847967267,
    "ASPIRATION_DELTA": 2908.7668884713516
}

CPP_CODE = """
#include <iostream>
#include <fstream>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <random>
#include <cstring>
#include <cstdlib>

extern "C" {

struct Board {
    int w, h, x;
    int col_height;
    uint64_t boards[2];
    int heights[7];
    uint64_t valid_mask;
    uint64_t zobrist_key;

    void from_kaggle(const int* kaggle_board, int my_mark, const uint64_t zobrist_table[2][49]) {
        w = 7; h = 6; x = 4;
        col_height = h + 1;
        boards[0] = 0; boards[1] = 0;
        zobrist_key = 0;
        int opp_mark = 3 - my_mark;

        for (int c = 0; c < w; ++c) {
            for (int r = 0; r < h; ++r) {
                int kaggle_idx = (h - 1 - r) * w + c;
                int val = kaggle_board[kaggle_idx];
                if (val != 0) {
                    int player_id = (val == my_mark) ? 0 : 1;
                    int idx = c * col_height + r;
                    boards[player_id] |= (1ULL << idx);
                    zobrist_key ^= zobrist_table[player_id][idx];
                }
            }
        }
        valid_mask = 0;
        for (int c = 0; c < w; ++c) {
            for (int r = 0; r < h; ++r) {
                valid_mask |= (1ULL << (c * col_height + r));
            }
        }
        for (int c = 0; c < w; ++c) {
            heights[c] = c * col_height + h;
            for (int r = 0; r < h; ++r) {
                int idx = c * col_height + r;
                uint64_t mask = 1ULL << idx;
                if (!(boards[0] & mask) && !(boards[1] & mask)) {
                    heights[c] = c * col_height + r;
                    break;
                }
            }
        }
    }

    inline void make_move(int col, int player_id, const uint64_t zobrist_table[2][49], uint64_t zobrist_turn) {
        int idx = heights[col];
        boards[player_id] |= (1ULL << idx);
        zobrist_key ^= zobrist_table[player_id][idx];
        zobrist_key ^= zobrist_turn;
        heights[col]++;
    }

    inline void undo_move(int col, int player_id, const uint64_t zobrist_table[2][49], uint64_t zobrist_turn) {
        heights[col]--;
        int idx = heights[col];
        boards[player_id] &= ~(1ULL << idx);
        zobrist_key ^= zobrist_table[player_id][idx];
        zobrist_key ^= zobrist_turn;
    }

    inline bool check_win(int player_id) const {
        uint64_t b = boards[player_id];
        int shifts[4] = {1, col_height, col_height + 1, col_height - 1};
        for (int s : shifts) {
            uint64_t filled = b;
            int len_filled = 1;
            while (len_filled < x) {
                int delta = std::min(len_filled, x - len_filled);
                filled &= (filled >> (delta * s));
                len_filled += delta;
            }
            if (filled != 0) return true;
        }
        return false;
    }

    inline int get_valid_cols(int* out_cols) const {
        int count = 0;
        for (int c = 0; c < w; ++c) {
            if (heights[c] < (c * col_height + h)) {
                out_cols[count++] = c;
            }
        }
        return count;
    }
};

struct TTEntry {
    uint64_t key;
    int32_t score;
    int16_t depth;
    int8_t flag;
    int8_t best_move;
};

const int TT_SIZE = 1 << 23; 
TTEntry tt[TT_SIZE]; 

struct BookTable {
    uint32_t size;
    int key_bytes;
    void* keys;
    uint8_t* values;

    BookTable() : size(0), key_bytes(0), keys(nullptr), values(nullptr) {}

    ~BookTable() {
        if (keys) free(keys);
        if (values) free(values);
    }

    uint8_t get(uint64_t key) const {
        if (size == 0) return 0;
        uint32_t idx = key % size;
        
        // Quét tuyến tính (Linear Probing) chuẩn cấu trúc Pascal Pons Solver
        if (key_bytes == 1) {
            uint8_t* k_arr = (uint8_t*)keys;
            uint8_t target = (uint8_t)key;
            while (k_arr[idx] != 0) {
                if (k_arr[idx] == target) return values[idx];
                idx = (idx + 1) % size;
            }
        } else if (key_bytes == 2) {
            uint16_t* k_arr = (uint16_t*)keys;
            uint16_t target = (uint16_t)key;
            while (k_arr[idx] != 0) {
                if (k_arr[idx] == target) return values[idx];
                idx = (idx + 1) % size;
            }
        } else if (key_bytes == 4) {
            uint32_t* k_arr = (uint32_t*)keys;
            uint32_t target = (uint32_t)key;
            while (k_arr[idx] != 0) {
                if (k_arr[idx] == target) return values[idx];
                idx = (idx + 1) % size;
            }
        }
        return 0; 
    }
};

BookTable book_table;
int book_depth = -1;

bool is_prime(uint32_t n) {
    if (n <= 1) return false;
    if (n <= 3) return true;
    if (n % 2 == 0 || n % 3 == 0) return false;
    for (uint32_t i = 5; i * i <= n; i += 6) {
        if (n % i == 0 || n % (i + 2) == 0) return false;
    }
    return true;
}

uint32_t next_prime(uint32_t n) {
    while (!is_prime(n)) n++;
    return n;
}

void load_opening_book(const char* filepath) {
    std::ifstream ifs(filepath, std::ios::binary);
    if (ifs.fail()) {
        std::cerr << "Unable to load opening book: " << filepath << std::endl;
        return;
    }

    char width, height, depth, partial_key_bytes, value_bytes, log_size;
    ifs.read(&width, 1);
    ifs.read(&height, 1);
    ifs.read(&depth, 1);
    ifs.read(&partial_key_bytes, 1);
    ifs.read(&value_bytes, 1);
    ifs.read(&log_size, 1);

    uint32_t raw_size = 1 << (int)log_size;
    uint32_t size = next_prime(raw_size);

    book_table.size = size;
    book_table.key_bytes = (int)partial_key_bytes;
    book_table.keys = malloc(size * book_table.key_bytes);
    book_table.values = (uint8_t*)malloc(size * 1);

    ifs.read(reinterpret_cast<char*>(book_table.keys), size * book_table.key_bytes);
    ifs.read(reinterpret_cast<char*>(book_table.values), size * 1);
    
    book_depth = (int)depth;
    std::cerr << "Successfully loaded opening book. Max Depth: " << book_depth << std::endl;
    ifs.close();
}

struct Evaluator {
    double win_base, fork_score, threat_score, max_strat;
    double c_smooth, k_edge, k_corner, cnn_power;
    double alpha_balanced, alpha_defensive;
    double my_edge_pow, my_corner_pow, my_both_pow;

    uint64_t center_mask, flank_mask, bottom_mask, even_rows_mask, odd_rows_mask;
    bool initialized = false;

    void init(const double* w_arr) {
        win_base = w_arr[0];
        fork_score = w_arr[1];
        threat_score = w_arr[2];
        max_strat = w_arr[3];
        c_smooth = w_arr[4];
        k_edge = w_arr[5];
        k_corner = w_arr[6];
        cnn_power = w_arr[7];
        alpha_balanced = w_arr[8];
        alpha_defensive = w_arr[9];

        my_edge_pow = std::pow(k_edge, cnn_power);
        my_corner_pow = std::pow(k_corner, cnn_power);
        my_both_pow = std::pow(k_edge + k_corner, cnn_power);

        int col_height = 7;
        int center_col = 3;
        center_mask = ((1ULL << 6) - 1) << (center_col * col_height);

        flank_mask = 0;
        flank_mask |= ((1ULL << 6) - 1) << (2 * col_height);
        flank_mask |= ((1ULL << 6) - 1) << (4 * col_height);

        bottom_mask = 0;
        even_rows_mask = 0;
        odd_rows_mask = 0;
        for (int c = 0; c < 7; ++c) {
            bottom_mask |= (1ULL << (c * col_height));
            for (int r = 0; r < 6; r += 2) {
                even_rows_mask |= (1ULL << (c * col_height + r));
            }
            for (int r = 1; r < 6; r += 2) {
                odd_rows_mask |= (1ULL << (c * col_height + r));
            }
        }
        initialized = true;
    }

    inline uint64_t get_winning_spots(uint64_t P, const int* shifts, uint64_t valid_mask) const {
        uint64_t threats = 0;
        for (int i = 0; i < 4; ++i) {
            int s = shifts[i];
            uint64_t p_right2 = (P >> s) & (P >> (2 * s));
            uint64_t p_left2 = (P << s) & (P << (2 * s));
            threats |= p_right2 & ((P >> (3 * s)) | (P << s));
            threats |= p_left2 & ((P << (3 * s)) | (P >> s));
        }
        return threats & valid_mask;
    }

    int evaluate(const Board& board, int current_player_id) const {
        int my_id = current_player_id;
        int opp_id = 1 - my_id;
        uint64_t my_board = board.boards[my_id];
        uint64_t opp_board = board.boards[opp_id];
        uint64_t occupied = my_board | opp_board;

        uint64_t playable_mask = (occupied + bottom_mask) & board.valid_mask;
        int shifts[4] = {1, 7, 8, 6};
        uint64_t my_all_winning_spots = get_winning_spots(my_board, shifts, board.valid_mask);
        uint64_t opp_all_winning_spots = get_winning_spots(opp_board, shifts, board.valid_mask);

        uint64_t my_immediate_threats = my_all_winning_spots & playable_mask;
        uint64_t opp_immediate_threats = opp_all_winning_spots & playable_mask;

        int my_threats_count = __builtin_popcountll(my_immediate_threats);
        int opp_threats_count = __builtin_popcountll(opp_immediate_threats);

        if (opp_threats_count >= 2) return -fork_score;
        if (my_threats_count >= 2) return fork_score;

        uint64_t my_parity_mask = (my_id == 0) ? even_rows_mask : odd_rows_mask;
        uint64_t opp_parity_mask = (opp_id == 0) ? even_rows_mask : odd_rows_mask;
        int my_parity_threats = __builtin_popcountll(my_immediate_threats & my_parity_mask);
        int opp_parity_threats = __builtin_popcountll(opp_immediate_threats & opp_parity_mask);

        int my_center_bits = __builtin_popcountll(my_board & center_mask);
        int opp_center_bits = __builtin_popcountll(opp_board & center_mask);
        int my_flank_bits = __builtin_popcountll(my_board & flank_mask);
        int opp_flank_bits = __builtin_popcountll(opp_board & flank_mask);
        int center_score = (my_center_bits - opp_center_bits) * 150 + (my_flank_bits - opp_flank_bits) * 50;

        uint64_t empty_mask = (~occupied) & board.valid_mask;
        uint64_t my_edges_mask = 0;
        for (int i = 0; i < 2; ++i) {
            int s = shifts[i];
            my_edges_mask |= ((my_board << s) & board.valid_mask) | ((my_board >> s) & board.valid_mask);
        }
        uint64_t my_corners_mask = 0;
        for (int i = 2; i < 4; ++i) {
            int s = shifts[i];
            my_corners_mask |= ((my_board << s) & board.valid_mask) | ((my_board >> s) & board.valid_mask);
        }

        uint64_t opp_edges_mask = 0;
        for (int i = 0; i < 2; ++i) {
            int s = shifts[i];
            opp_edges_mask |= ((opp_board << s) & board.valid_mask) | ((opp_board >> s) & board.valid_mask);
        }
        uint64_t opp_corners_mask = 0;
        for (int i = 2; i < 4; ++i) {
            int s = shifts[i];
            opp_corners_mask |= ((opp_board << s) & board.valid_mask) | ((opp_board >> s) & board.valid_mask);
        }

        my_edges_mask &= empty_mask;
        my_corners_mask &= empty_mask;
        opp_edges_mask &= empty_mask;
        opp_corners_mask &= empty_mask;

        double total_my_cnn_score = (
            (__builtin_popcountll(my_edges_mask & ~my_corners_mask) * my_edge_pow) +
            (__builtin_popcountll(my_corners_mask & ~my_edges_mask) * my_corner_pow) +
            (__builtin_popcountll(my_edges_mask & my_corners_mask) * my_both_pow)
        );
        double total_opp_cnn_score = (
            (__builtin_popcountll(opp_edges_mask & ~opp_corners_mask) * my_edge_pow) +
            (__builtin_popcountll(opp_corners_mask & ~opp_edges_mask) * my_corner_pow) +
            (__builtin_popcountll(opp_edges_mask & opp_corners_mask) * my_both_pow)
        );

        double my_soft = total_my_cnn_score > 0 ? total_my_cnn_score / (total_my_cnn_score + c_smooth) : 0;
        double opp_soft = total_opp_cnn_score > 0 ? total_opp_cnn_score / (total_opp_cnn_score + c_smooth) : 0;
        double alpha = (total_my_cnn_score >= total_opp_cnn_score) ? alpha_balanced : alpha_defensive;

        double strategic_score = (max_strat * my_soft) - (max_strat * opp_soft * alpha);
        double parity_score = (my_parity_threats * threat_score) - (opp_parity_threats * (threat_score * 1.2));

        return (int)(strategic_score + parity_score + center_score);
    }
};

struct Searcher {
    Evaluator eval;
    uint64_t zobrist_table[2][49];
    uint64_t zobrist_turn;
    int killer_moves[64][2];
    int history_table[7];

    std::chrono::steady_clock::time_point start_time;
    double time_limit;
    bool is_timeout;
    int node_count;

    void init_search(const double* weights) {
        eval.init(weights);
        std::mt19937_64 rng(42);
        for (int p = 0; p < 2; ++p) {
            for (int i = 0; i < 49; ++i) {
                zobrist_table[p][i] = rng();
            }
        }
        zobrist_turn = rng();
        std::memset(history_table, 0, sizeof(history_table));
        std::memset(killer_moves, 0, sizeof(killer_moves));
    }

    inline void check_signals() {
        node_count++;
        if ((node_count & 1023) == 0) {
            auto now = std::chrono::steady_clock::now();
            double elapsed = std::chrono::duration<double>(now - start_time).count();
            if (elapsed > time_limit) {
                is_timeout = true;
            }
        }
    }

    void order_moves(const Board& board, int* cols, int count, int tt_move, int ply, int last_move) {
        int scores[7];
        int k1 = killer_moves[ply][0];
        int k2 = killer_moves[ply][1];
        int target_center = (last_move != -1) ? last_move : 3;

        for (int i = 0; i < count; ++i) {
            int col = cols[i];
            int score = 0;
            if (col == tt_move) score += 1000000;
            else if (col == k1) score += 500000;
            else if (col == k2) score += 250000;
            score += history_table[col] / 100;
            score -= std::abs(col - target_center) * 100;
            scores[i] = score;
        }

        for (int i = 1; i < count; ++i) {
            int temp_col = cols[i];
            int temp_score = scores[i];
            int j = i - 1;
            while (j >= 0 && scores[j] < temp_score) {
                cols[j + 1] = cols[j];
                scores[j + 1] = scores[j];
                j--;
            }
            cols[j + 1] = temp_col;
            scores[j + 1] = temp_score;
        }
    }

    int quiesce(Board& board, int alpha, int beta, int current_player_id, int ply) {
        check_signals();
        if (is_timeout) return 0;

        int stand_pat = eval.evaluate(board, current_player_id);
        if (stand_pat >= beta) return stand_pat;
        alpha = std::max(alpha, stand_pat);

        int cols[7];
        int count = board.get_valid_cols(cols);
        int opp_id = 1 - current_player_id;

        for (int i = 0; i < count; ++i) {
            int col = cols[i];
            board.make_move(col, current_player_id, zobrist_table, zobrist_turn);
            bool is_win = board.check_win(current_player_id);
            board.undo_move(col, current_player_id, zobrist_table, zobrist_turn);
            if (is_win) {
                return eval.win_base - (ply + 1);
            }
        }

        int forced_count = 0;
        int forced_col = -1;
        for (int i = 0; i < count; ++i) {
            int col = cols[i];
            board.make_move(col, opp_id, zobrist_table, zobrist_turn);
            bool is_win = board.check_win(opp_id);
            board.undo_move(col, opp_id, zobrist_table, zobrist_turn);
            if (is_win) {
                forced_count++;
                forced_col = col;
            }
        }

        if (forced_count > 1) {
            return -eval.win_base + (ply + 2);
        }

        if (forced_count == 1) {
            board.make_move(forced_col, current_player_id, zobrist_table, zobrist_turn);
            int score = -quiesce(board, -beta, -alpha, opp_id, ply + 1);
            board.undo_move(forced_col, current_player_id, zobrist_table, zobrist_turn);
            if (is_timeout) return 0;
            if (score >= beta) return score;
            alpha = std::max(alpha, score);
        }

        return alpha;
    }

    int negamax(Board& board, int depth, int alpha, int beta, int current_player_id, int ply, int last_move) {
        int alpha_orig = alpha;
        check_signals();
        if (is_timeout) return 0;

        uint64_t key = board.zobrist_key;
        int tt_idx = key % TT_SIZE;
        TTEntry& entry = tt[tt_idx];
        int tt_move = -1;

        if (entry.key == key && entry.depth >= depth) {
            int score = entry.score;
            if (score > 9000000) score -= ply;
            else if (score < -9000000) score += ply;

            if (entry.flag == 0) return score; 
            else if (entry.flag == 1) alpha = std::max(alpha, score); 
            else if (entry.flag == 2) beta = std::min(beta, score); 
            if (alpha >= beta) return score;
        }

        if (entry.key == key) {
            tt_move = entry.best_move;
        }

        int opp_id = 1 - current_player_id;
        if (board.check_win(opp_id)) return -eval.win_base + ply;
        if (board.check_win(current_player_id)) return eval.win_base - ply;

        int cols[7];
        int count = board.get_valid_cols(cols);
        if (depth == 0 || count == 0) {
            return quiesce(board, alpha, beta, current_player_id, ply);
        }

        order_moves(board, cols, count, tt_move, ply, last_move);

        int max_eval = -20000000;
        int best_move_found = cols[0];

        for (int i = 0; i < count; ++i) {
            int col = cols[i];
            board.make_move(col, current_player_id, zobrist_table, zobrist_turn);
            int score;
            if (i == 0) {
                score = -negamax(board, depth - 1, -beta, -alpha, opp_id, ply + 1, col);
            } else {
                score = -negamax(board, depth - 1, -alpha - 1, -alpha, opp_id, ply + 1, col);
                if (alpha < score && score < beta) {
                    score = -negamax(board, depth - 1, -beta, -score, opp_id, ply + 1, col);
                }
            }
            board.undo_move(col, current_player_id, zobrist_table, zobrist_turn);

            if (is_timeout) return 0;

            if (score > max_eval) {
                max_eval = score;
                best_move_found = col;
            }
            alpha = std::max(alpha, score);
            if (alpha >= beta) {
                if (col != tt_move) {
                    if (killer_moves[ply][0] != col) {
                        killer_moves[ply][1] = killer_moves[ply][0];
                        killer_moves[ply][0] = col;
                    }
                    history_table[col] += depth * depth;
                }
                break;
            }
        }

        if (!is_timeout) {
            int flag = 0; 
            if (max_eval <= alpha_orig) flag = 2; 
            else if (max_eval >= beta) flag = 1; 

            int tt_score = max_eval;
            if (max_eval > 9000000) tt_score += ply;
            else if (max_eval < -9000000) tt_score -= ply;

            tt[tt_idx] = {key, tt_score, (int16_t)depth, (int8_t)flag, (int8_t)best_move_found};
        }

        return max_eval;
    }
};

Searcher searcher;
bool engine_initialized = false;

void init_cpp_engine(const double* weights) {
    if (!engine_initialized) {
        searcher.init_search(weights);
        engine_initialized = true;
    }
}

int select_move_cpp(const int* kaggle_board, int my_mark, int max_depth, double time_limit) {
    Board board;
    board.from_kaggle(kaggle_board, my_mark, searcher.zobrist_table);

    searcher.start_time = std::chrono::steady_clock::now();
    searcher.time_limit = time_limit;
    searcher.is_timeout = false;
    searcher.node_count = 0;
    std::memset(searcher.history_table, 0, sizeof(searcher.history_table));

    int cols[7];
    int count = board.get_valid_cols(cols);
    if (count == 0) return -1;
    if (count == 1) return cols[0];

    if (book_depth != -1) {
        int best_book_col = -1;
        uint8_t min_opp_val = 255;

        for (int i = 0; i < count; ++i) {
            int col = cols[i];
            
            // Đi thử nước đi
            board.make_move(col, 0, searcher.zobrist_table, searcher.zobrist_turn);
            
            // Tính toán khóa băm hoán vị key3 của Pons cho thế cờ sau nước đi
            uint64_t next_pos = board.boards[1]; 
            uint64_t next_mask = board.boards[0] | board.boards[1];
            uint64_t key3 = next_pos + next_mask;
            
            // Tra cứu giá trị trong Opening Book
            uint8_t val = book_table.get(key3);
            
            // Hoàn tác nước đi
            board.undo_move(col, 0, searcher.zobrist_table, searcher.zobrist_turn);
            
            // Do đối thủ chuẩn bị đi, ta phải tìm nước đi để MINIMIZE điểm số của đối thủ
            if (val > 0 && val < min_opp_val) {
                min_opp_val = val;
                best_book_col = col;
            }
        }
        
        // Nếu tìm thấy nước đi hoàn hảo trong sách giáo khoa, đi ngay lập tức!
        if (best_book_col != -1) {
            return best_book_col;
        }
    }

    // Nếu không có sách giáo khoa (hoặc đã vượt quá Opening Book), gọi Negamax cực hạn
    int overall_best_col = cols[0];
    int last_depth_score = 0;
    int ASPIRATION_DELTA = 2908; 

    for (int current_depth = 1; current_depth <= max_depth; ++current_depth) {
        int alpha = -20000000;
        int beta = 20000000;
        if (current_depth >= 3) {
            alpha = last_depth_score - ASPIRATION_DELTA;
            beta = last_depth_score + ASPIRATION_DELTA;
        }

        while (true) {
            int current_alpha = alpha;
            int current_beta = beta;
            int best_score = -20000000;
            int best_col = -1;

            uint64_t key = board.zobrist_key;
            int tt_idx = key % TT_SIZE;
            int tt_move = (tt[tt_idx].key == key) ? tt[tt_idx].best_move : -1;

            int ordered_cols[7];
            std::copy(cols, cols + count, ordered_cols);
            searcher.order_moves(board, ordered_cols, count, tt_move, 1, -1);

            for (int i = 0; i < count; ++i) {
                int col = ordered_cols[i];
                board.make_move(col, 0, searcher.zobrist_table, searcher.zobrist_turn);
                int score;
                if (i == 0) {
                    score = -searcher.negamax(board, current_depth - 1, -current_beta, -current_alpha, 1, 1, col);
                } else {
                    score = -searcher.negamax(board, current_depth - 1, -current_alpha - 1, -current_alpha, 1, 1, col);
                    if (current_alpha < score && score < current_beta) {
                        score = -searcher.negamax(board, current_depth - 1, -current_beta, -score, 1, 1, col);
                    }
                }
                board.undo_move(col, 0, searcher.zobrist_table, searcher.zobrist_turn);

                if (searcher.is_timeout) break;

                if (score > best_score) {
                    best_score = score;
                    best_col = col;
                }
                current_alpha = std::max(current_alpha, score);
                if (current_alpha >= current_beta) break;
            }

            if (searcher.is_timeout) break;

            if (best_score <= alpha) {
                alpha = -20000000;
                continue;
            } else if (best_score >= beta) {
                beta = 20000000;
                continue;
            }
            
            if (best_col != -1) {
                overall_best_col = best_col;
                last_depth_score = best_score;
            }
            break;
        }

        if (searcher.is_timeout) break;
        if (last_depth_score >= 10000000 - 100) break;
    }

    return overall_best_col;
}

}
"""

_lib = None

def compile_and_load_cpp():
    """Tự động xuất tệp C++ ra đĩa và dùng g++ để tối ưu hóa cực hạn O3"""
    global _lib
    if _lib is not None:
        return

    cpp_filename = "/tmp/libengine.cpp"
    so_filename = "/tmp/libengine.so"

    with open(cpp_filename, "w") as f:
        f.write(CPP_CODE)

    compile_cmd = [
        "g++", "-O3", "-shared", "-std=c++17",
        "-march=native", "-funroll-loops", "-fPIC",
        cpp_filename, "-o", so_filename
    ]
    
    try:
        subprocess.run(compile_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as e:
        raise RuntimeError(f"❌ [LỖI BIÊN DỊCH C++] Không thể build engine tối ưu: {e}")

    _lib = ctypes.CDLL(so_filename)
    
    _lib.init_cpp_engine.argtypes = [ctypes.POINTER(ctypes.c_double)]
    _lib.init_cpp_engine.restype = None

    _lib.load_opening_book.argtypes = [ctypes.c_char_p]
    _lib.load_opening_book.restype = None

    _lib.select_move_cpp.argtypes = [
        ctypes.POINTER(ctypes.c_int), # mảng bàn cờ phẳng
        ctypes.c_int,                 # mark của bot
        ctypes.c_int,                 # max depth
        ctypes.c_double               # time limit
    ]
    _lib.select_move_cpp.restype = ctypes.c_int

    weights_list = [
        CHAMPION_WEIGHTS["WIN_BASE"],
        CHAMPION_WEIGHTS["FORK_SCORE"],
        CHAMPION_WEIGHTS["THREAT_SCORE"],
        CHAMPION_WEIGHTS["MAX_STRATEGIC"],
        CHAMPION_WEIGHTS["C_SMOOTH"],
        CHAMPION_WEIGHTS["K_EDGE"],
        CHAMPION_WEIGHTS["K_CORNER"],
        CHAMPION_WEIGHTS["CNN_POWER"],
        CHAMPION_WEIGHTS["ALPHA_BALANCED"],
        CHAMPION_WEIGHTS["ALPHA_DEFENSIVE"]
    ]
    weights_arr = (ctypes.c_double * len(weights_list))(*weights_list)
    
    _lib.init_cpp_engine(weights_arr)

    book_path = ""
    # Quét thư mục làm việc hiện tại
    for root, dirs, files in os.walk("."):
        for file in files:
            if file.endswith(".book"):
                book_path = os.path.join(root, file)
                break
                
    # Nếu không tìm thấy, quét thư mục đầu vào của Kaggle Dataset
    if not book_path and os.path.exists("/kaggle/input"):
        for root, dirs, files in os.walk("/kaggle/input"):
            for file in files:
                if file.endswith(".book"):
                    book_path = os.path.join(root, file)
                    break

    _lib.load_opening_book(book_path.encode('utf-8'))

def my_agent(observation, configuration):
    compile_and_load_cpp()
    
    # 1. Chuyển đổi mảng bàn cờ phẳng của Kaggle sang kiểu ctypes int*
    board_len = len(observation.board)
    c_board = (ctypes.c_int * board_len)(*observation.board)
    
    # 2. Cấu hình tìm kiếm cực hạn
    max_depth = 20
    time_limit = 2.2
    
    # 3. Gọi Engine C++ giải toán
    best_move = _lib.select_move_cpp(
        c_board,
        ctypes.c_int(observation.mark),
        ctypes.c_int(max_depth),
        ctypes.c_double(time_limit)
    )
    
    return int(best_move)