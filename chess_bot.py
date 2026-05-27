"""
=============================================================
  CHESS BOT - Python AI Chess Game
=============================================================

HOW TO RUN:
  1. Install: pip install python-chess
  2. Run:     python chess_bot.py

WHAT THIS FILE CONTAINS:
  Part 1 - Piece Values & Position Tables  (AI brain: values)
  Part 2 - Board Evaluation Function       (AI brain: scoring)
  Part 3 - Minimax + Alpha-Beta Pruning   (AI brain: thinking)
  Part 4 - GUI (Visual Chess Board)        (What you see)
  Part 5 - Main Entry Point               (Starts the game)
=============================================================
"""

import tkinter as tk
from tkinter import messagebox
import chess          # python-chess library handles all chess rules
import random


# =============================================================
# PART 1 — PIECE VALUES & POSITION TABLES
# =============================================================
# These numbers tell the AI how valuable each piece is.
# Higher value = more important piece.

PIECE_VALUES = {
    chess.PAWN:   100,    # Pawns are worth 1 point
    chess.KNIGHT: 320,    # Knights are worth ~3 points
    chess.BISHOP: 330,    # Bishops are worth ~3.3 points
    chess.ROOK:   500,    # Rooks are worth 5 points
    chess.QUEEN:  900,    # Queen is worth 9 points
    chess.KING:   20000,  # King is priceless (can't lose it!)
}

# POSITION TABLES — bonus points for placing pieces on good squares.
# Each table is a list of 64 numbers (one per square on the board).
# Positive = good square, Negative = bad square for that piece.

# Pawns should advance toward the opponent's side
PAWN_TABLE = [
     0,  0,  0,  0,  0,  0,  0,  0,
    50, 50, 50, 50, 50, 50, 50, 50,
    10, 10, 20, 30, 30, 20, 10, 10,
     5,  5, 10, 25, 25, 10,  5,  5,
     0,  0,  0, 20, 20,  0,  0,  0,
     5, -5,-10,  0,  0,-10, -5,  5,
     5, 10, 10,-20,-20, 10, 10,  5,
     0,  0,  0,  0,  0,  0,  0,  0,
]

# Knights prefer the center of the board
KNIGHT_TABLE = [
    -50,-40,-30,-30,-30,-30,-40,-50,
    -40,-20,  0,  0,  0,  0,-20,-40,
    -30,  0, 10, 15, 15, 10,  0,-30,
    -30,  5, 15, 20, 20, 15,  5,-30,
    -30,  0, 15, 20, 20, 15,  0,-30,
    -30,  5, 10, 15, 15, 10,  5,-30,
    -40,-20,  0,  5,  5,  0,-20,-40,
    -50,-40,-30,-30,-30,-30,-40,-50,
]

# Bishops like open diagonals
BISHOP_TABLE = [
    -20,-10,-10,-10,-10,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5, 10, 10,  5,  0,-10,
    -10,  5,  5, 10, 10,  5,  5,-10,
    -10,  0, 10, 10, 10, 10,  0,-10,
    -10, 10, 10, 10, 10, 10, 10,-10,
    -10,  5,  0,  0,  0,  0,  5,-10,
    -20,-10,-10,-10,-10,-10,-10,-20,
]

# Rooks like open files (columns with no pawns)
ROOK_TABLE = [
     0,  0,  0,  0,  0,  0,  0,  0,
     5, 10, 10, 10, 10, 10, 10,  5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
     0,  0,  0,  5,  5,  0,  0,  0,
]

# Queen combines rook and bishop strengths
QUEEN_TABLE = [
    -20,-10,-10, -5, -5,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5,  5,  5,  5,  0,-10,
     -5,  0,  5,  5,  5,  5,  0, -5,
      0,  0,  5,  5,  5,  5,  0, -5,
    -10,  5,  5,  5,  5,  5,  0,-10,
    -10,  0,  5,  0,  0,  0,  0,-10,
    -20,-10,-10, -5, -5,-10,-10,-20,
]

# King should stay safe (castle!) during the game
KING_TABLE = [
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -20,-30,-30,-40,-40,-30,-30,-20,
    -10,-20,-20,-20,-20,-20,-20,-10,
     20, 20,  0,  0,  0,  0, 20, 20,
     20, 30, 10,  0,  0, 10, 30, 20,
]

# Group all tables for easy lookup
PIECE_TABLES = {
    chess.PAWN:   PAWN_TABLE,
    chess.KNIGHT: KNIGHT_TABLE,
    chess.BISHOP: BISHOP_TABLE,
    chess.ROOK:   ROOK_TABLE,
    chess.QUEEN:  QUEEN_TABLE,
    chess.KING:   KING_TABLE,
}


def get_position_bonus(piece_type, square, color):
    """
    Get the bonus score for placing a piece on a specific square.
    White pieces look at the board from the bottom (rank 1).
    Black pieces look at the board from the top (rank 8).
    """
    table = PIECE_TABLES[piece_type]
    rank = chess.square_rank(square)  # 0 = rank 1 (bottom), 7 = rank 8 (top)
    file = chess.square_file(square)  # 0 = a-file, 7 = h-file

    if color == chess.WHITE:
        # White sees rank 1 as the near side → flip vertically
        index = (7 - rank) * 8 + file
    else:
        # Black sees rank 8 as the near side → no flip
        index = rank * 8 + file

    return table[index]


# =============================================================
# PART 2 — BOARD EVALUATION FUNCTION
# =============================================================
# This function assigns a single number to any board position.
#   Positive number  → good for White (human player)
#   Negative number  → good for Black (AI)
# The AI will always try to pick moves that make this number
# as small (negative) as possible.

def evaluate_board(board):
    """
    Score the board from White's perspective.
    White wants high scores. Black (AI) wants low scores.
    """
    # Special game-over cases
    if board.is_checkmate():
        # Whoever's turn it is just got checkmated → they lose
        return -99999 if board.turn == chess.WHITE else 99999

    if board.is_stalemate() or board.is_insufficient_material():
        return 0  # Draw = neutral score

    score = 0

    # Loop through all 64 squares and add up piece values + position bonuses
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is None:
            continue  # Empty square, skip

        piece_value    = PIECE_VALUES[piece.piece_type]
        position_bonus = get_position_bonus(piece.piece_type, square, piece.color)
        total          = piece_value + position_bonus

        if piece.color == chess.WHITE:
            score += total   # White pieces ADD to score
        else:
            score -= total   # Black pieces SUBTRACT from score

    return score


# =============================================================
# PART 3 — MINIMAX ALGORITHM WITH ALPHA-BETA PRUNING
# =============================================================
# This is the "brain" of the chess bot.
#
# MINIMAX IDEA:
#   - The AI imagines all possible moves, then all opponent replies,
#     then all AI replies to those, etc. — forming a game tree.
#   - White (human) tries to MAXIMIZE the score.
#   - Black (AI)   tries to MINIMIZE the score.
#   - At the bottom of the tree, we evaluate the board.
#   - Then we "bubble up" the best scores.
#
# ALPHA-BETA PRUNING:
#   - Skips branches we already know won't affect the final result.
#   - Makes the search much faster — same result, less work!

def minimax(board, depth, alpha, beta, is_maximizing):
    """
    Recursively find the best move score.

    Parameters:
      board          - current chess board state
      depth          - how many more levels to search (0 = stop)
      alpha          - best score White can guarantee (starts at -∞)
      beta           - best score Black can guarantee (starts at +∞)
      is_maximizing  - True if it's White's turn, False for Black
    """
    # Base case: reached the end of our search depth or game is over
    if depth == 0 or board.is_game_over():
        return evaluate_board(board)

    if is_maximizing:
        # White's turn: pick move with highest score
        best_score = float('-inf')
        for move in board.legal_moves:
            board.push(move)                                      # Try move
            score = minimax(board, depth - 1, alpha, beta, False) # Recurse
            board.pop()                                           # Undo move
            best_score = max(best_score, score)
            alpha = max(alpha, score)
            if beta <= alpha:
                break  # ✂ Beta cutoff: Black would never allow this
        return best_score
    else:
        # Black's (AI's) turn: pick move with lowest score
        best_score = float('inf')
        for move in board.legal_moves:
            board.push(move)                                     # Try move
            score = minimax(board, depth - 1, alpha, beta, True) # Recurse
            board.pop()                                          # Undo move
            best_score = min(best_score, score)
            beta = min(beta, score)
            if beta <= alpha:
                break  # ✂ Alpha cutoff: White would never allow this
        return best_score


def get_best_move(board, depth=3):
    """
    Find the AI's best move using minimax.
    The AI plays as Black (minimizing player).
    """
    best_move  = None
    best_score = float('inf')   # Black wants the lowest possible score

    # Shuffle moves so the AI plays different games when scores are equal
    moves = list(board.legal_moves)
    random.shuffle(moves)

    for move in moves:
        board.push(move)
        score = minimax(board, depth - 1, float('-inf'), float('inf'), True)
        board.pop()

        if score < best_score:
            best_score = score
            best_move  = move

    return best_move


# =============================================================
# PART 4 — GUI (Graphical Chess Board using Tkinter)
# =============================================================

# Unicode chess piece symbols — shown on the board
PIECE_SYMBOLS = {
    (chess.KING,   chess.WHITE): "♔",
    (chess.QUEEN,  chess.WHITE): "♕",
    (chess.ROOK,   chess.WHITE): "♖",
    (chess.BISHOP, chess.WHITE): "♗",
    (chess.KNIGHT, chess.WHITE): "♘",
    (chess.PAWN,   chess.WHITE): "♙",
    (chess.KING,   chess.BLACK): "♚",
    (chess.QUEEN,  chess.BLACK): "♛",
    (chess.ROOK,   chess.BLACK): "♜",
    (chess.BISHOP, chess.BLACK): "♝",
    (chess.KNIGHT, chess.BLACK): "♞",
    (chess.PAWN,   chess.BLACK): "♟",
}


class ChessBotApp:
    """
    The main application class.
    Handles the window, board drawing, and player/AI moves.
    """

    # --- Color palette ---
    COLOR = {
        "bg":          "#1E1E2E",   # Dark background
        "panel":       "#2A2A3E",   # Side panel background
        "light_sq":    "#F0D9B5",   # Beige light square
        "dark_sq":     "#B58863",   # Brown dark square
        "selected":    "#6BCB77",   # Green: selected piece
        "legal":       "#4D96FF",   # Blue: legal move target
        "last_move":   "#D4E157",   # Yellow: last move highlight
        "check":       "#EF5350",   # Red: king in check
        "text":        "#FFFFFF",
        "subtext":     "#AAAACC",
        "gold":        "#FFD700",
        "btn_new":     "#4CAF50",
        "btn_hint":    "#2196F3",
        "btn_undo":    "#FF9800",
    }

    SQUARE_SIZE = 72   # pixels per square
    BOARD_SIZE  = SQUARE_SIZE * 8

    def __init__(self, root):
        self.root = root
        self.root.title("♟ Chess Bot — Python AI")
        self.root.configure(bg=self.COLOR["bg"])
        self.root.resizable(False, False)

        # Game state
        self.board        = chess.Board()
        self.selected_sq  = None         # Which square the player clicked
        self.legal_targets = []          # Where the selected piece can go
        self.last_move    = None         # Highlight last move
        self.game_over    = False
        self.ai_depth     = 3            # AI search depth (difficulty)
        self.move_history = []           # List of SAN move strings

        self._build_ui()
        self._draw_board()

    # ----------------------------------------------------------
    # UI CONSTRUCTION
    # ----------------------------------------------------------

    def _build_ui(self):
        """Build all widgets."""
        # ── Title bar ──────────────────────────────────────────
        title_bar = tk.Frame(self.root, bg=self.COLOR["bg"])
        title_bar.pack(pady=(14, 4))

        tk.Label(
            title_bar, text="♟  CHESS BOT",
            font=("Courier", 22, "bold"),
            bg=self.COLOR["bg"], fg=self.COLOR["gold"],
        ).pack()
        tk.Label(
            title_bar, text="Python AI · Minimax with Alpha-Beta Pruning",
            font=("Courier", 9),
            bg=self.COLOR["bg"], fg=self.COLOR["subtext"],
        ).pack()

        # ── Main row: board + side panel ──────────────────────
        main_row = tk.Frame(self.root, bg=self.COLOR["bg"])
        main_row.pack(padx=16, pady=8)

        self._build_board_section(main_row)
        self._build_side_panel(main_row)

    def _build_board_section(self, parent):
        """Build the board canvas with coordinate labels."""
        board_col = tk.Frame(parent, bg=self.COLOR["bg"])
        board_col.pack(side=tk.LEFT)

        # Top file labels (a – h)
        self._file_labels(board_col)

        board_row = tk.Frame(board_col, bg=self.COLOR["bg"])
        board_row.pack()

        # Left rank labels (8 – 1)
        self._rank_labels(board_row)

        # Canvas
        self.canvas = tk.Canvas(
            board_row,
            width=self.BOARD_SIZE, height=self.BOARD_SIZE,
            highlightthickness=3, highlightbackground=self.COLOR["gold"],
        )
        self.canvas.pack(side=tk.LEFT)
        self.canvas.bind("<Button-1>", self._on_click)

        # Bottom file labels
        self._file_labels(board_col)

        # Status label under board
        self.status_var = tk.StringVar(value="Your turn!  You are White  ♙")
        tk.Label(
            board_col, textvariable=self.status_var,
            font=("Courier", 12, "bold"),
            bg=self.COLOR["bg"], fg=self.COLOR["gold"],
            pady=6,
        ).pack()

    def _file_labels(self, parent):
        row = tk.Frame(parent, bg=self.COLOR["bg"])
        row.pack()
        tk.Label(row, text="   ", bg=self.COLOR["bg"]).pack(side=tk.LEFT)
        for f in "abcdefgh":
            tk.Label(
                row, text=f, width=int(self.SQUARE_SIZE / 10),
                font=("Courier", 10), bg=self.COLOR["bg"], fg=self.COLOR["subtext"],
            ).pack(side=tk.LEFT)

    def _rank_labels(self, parent):
        col = tk.Frame(parent, bg=self.COLOR["bg"])
        col.pack(side=tk.LEFT)
        for r in range(8, 0, -1):
            tk.Label(
                col, text=str(r),
                font=("Courier", 10), bg=self.COLOR["bg"], fg=self.COLOR["subtext"],
                width=2, height=int(self.SQUARE_SIZE / 14),
            ).pack()

    def _build_side_panel(self, parent):
        """Build the right-side controls and move history."""
        panel = tk.Frame(parent, bg=self.COLOR["panel"], padx=14, pady=14)
        panel.pack(side=tk.LEFT, padx=(14, 0), fill=tk.Y)

        def section_label(text):
            tk.Label(
                panel, text=text,
                font=("Courier", 9, "bold"),
                bg=self.COLOR["panel"], fg=self.COLOR["subtext"],
            ).pack(anchor=tk.W, pady=(10, 2))

        # ── Game info ─────────────────────────────────────────
        section_label("GAME INFO")
        self.move_num_var = tk.StringVar(value="Move:  1")
        tk.Label(
            panel, textvariable=self.move_num_var,
            font=("Courier", 11, "bold"),
            bg=self.COLOR["panel"], fg=self.COLOR["text"],
        ).pack(anchor=tk.W)

        # ── Difficulty ────────────────────────────────────────
        section_label("AI DIFFICULTY")
        diff_frame = tk.Frame(panel, bg=self.COLOR["panel"])
        diff_frame.pack(anchor=tk.W)
        self.diff_var = tk.StringVar(value="Medium")
        for label, depth in [("Easy", 1), ("Medium", 3), ("Hard", 4)]:
            tk.Radiobutton(
                diff_frame, text=label, variable=self.diff_var, value=label,
                command=lambda d=depth: setattr(self, 'ai_depth', d),
                font=("Courier", 10),
                bg=self.COLOR["panel"], fg=self.COLOR["text"],
                selectcolor=self.COLOR["bg"],
                activebackground=self.COLOR["panel"],
            ).pack(anchor=tk.W)

        # ── Buttons ───────────────────────────────────────────
        section_label("CONTROLS")

        def make_btn(text, cmd, color):
            tk.Button(
                panel, text=text, command=cmd,
                font=("Courier", 10, "bold"),
                bg=color, fg="white", relief=tk.FLAT,
                padx=10, pady=5, cursor="hand2", width=14,
            ).pack(pady=3)

        make_btn("🔄  New Game", self._new_game,  self.COLOR["btn_new"])
        make_btn("💡  Hint",     self._show_hint, self.COLOR["btn_hint"])
        make_btn("↩  Undo",     self._undo_move, self.COLOR["btn_undo"])

        # ── Move history ──────────────────────────────────────
        section_label("MOVE HISTORY")
        history_frame = tk.Frame(panel, bg="#16162A", relief=tk.SUNKEN, bd=1)
        history_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 4))

        scrollbar = tk.Scrollbar(history_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.history_box = tk.Text(
            history_frame, width=16, height=14,
            font=("Courier", 9),
            bg="#16162A", fg=self.COLOR["text"],
            relief=tk.FLAT, state=tk.DISABLED,
            yscrollcommand=scrollbar.set,
        )
        self.history_box.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=4, pady=4)
        scrollbar.config(command=self.history_box.yview)

        # ── Legend ────────────────────────────────────────────
        section_label("LEGEND")
        for color_name, desc in [
            (self.COLOR["selected"],  "Selected"),
            (self.COLOR["legal"],     "Legal move"),
            (self.COLOR["last_move"], "Last move"),
            (self.COLOR["check"],     "Check!"),
        ]:
            row = tk.Frame(panel, bg=self.COLOR["panel"])
            row.pack(anchor=tk.W, pady=1)
            tk.Label(row, text="■", fg=color_name, bg=self.COLOR["panel"],
                     font=("Arial", 12)).pack(side=tk.LEFT)
            tk.Label(row, text=f" {desc}", fg=self.COLOR["subtext"],
                     bg=self.COLOR["panel"], font=("Courier", 9)).pack(side=tk.LEFT)

    # ----------------------------------------------------------
    # BOARD DRAWING
    # ----------------------------------------------------------

    def _draw_board(self):
        """Redraw the entire chess board."""
        self.canvas.delete("all")

        for sq in chess.SQUARES:
            file = chess.square_file(sq)
            rank = chess.square_rank(sq)
            x    = file * self.SQUARE_SIZE
            y    = (7 - rank) * self.SQUARE_SIZE

            # Determine base color
            is_light = (file + rank) % 2 == 0
            color = self.COLOR["light_sq"] if is_light else self.COLOR["dark_sq"]

            # Highlight last move squares
            if self.last_move and sq in (self.last_move.from_square, self.last_move.to_square):
                color = self.COLOR["last_move"]

            # Highlight selected piece square
            if sq == self.selected_sq:
                color = self.COLOR["selected"]

            # Highlight legal move targets
            if sq in self.legal_targets:
                color = self.COLOR["legal"]

            # Highlight king if in check
            if self.board.is_check():
                king_sq = self.board.king(self.board.turn)
                if sq == king_sq:
                    color = self.COLOR["check"]

            # Draw square
            self.canvas.create_rectangle(
                x, y, x + self.SQUARE_SIZE, y + self.SQUARE_SIZE,
                fill=color, outline="",
            )

            # Draw piece (if any)
            piece = self.board.piece_at(sq)
            if piece:
                symbol   = PIECE_SYMBOLS[(piece.piece_type, piece.color)]
                cx, cy   = x + self.SQUARE_SIZE // 2, y + self.SQUARE_SIZE // 2
                piece_fg = "white" if piece.color == chess.WHITE else "#111111"
                # Drop shadow
                self.canvas.create_text(cx+2, cy+2, text=symbol,
                                        font=("Arial", 32), fill="#555555")
                # Piece
                self.canvas.create_text(cx, cy, text=symbol,
                                        font=("Arial", 32, "bold"), fill=piece_fg)

            # Draw a dot on legal move squares that are empty
            if sq in self.legal_targets and not self.board.piece_at(sq):
                cx, cy = x + self.SQUARE_SIZE // 2, y + self.SQUARE_SIZE // 2
                r = 10
                self.canvas.create_oval(cx-r, cy-r, cx+r, cy+r,
                                        fill="#33333366", outline="")

    # ----------------------------------------------------------
    # EVENT HANDLERS
    # ----------------------------------------------------------

    def _on_click(self, event):
        """Handle player clicking a square."""
        if self.game_over or self.board.turn != chess.WHITE:
            return  # Not player's turn

        file = event.x // self.SQUARE_SIZE
        rank = 7 - (event.y // self.SQUARE_SIZE)
        if not (0 <= file <= 7 and 0 <= rank <= 7):
            return

        sq    = chess.square(file, rank)
        piece = self.board.piece_at(sq)

        if self.selected_sq is not None:
            # ── We already have a piece selected ──────────────
            move = chess.Move(self.selected_sq, sq)

            # Pawn promotion → auto-promote to Queen
            selected_piece = self.board.piece_at(self.selected_sq)
            if (selected_piece
                    and selected_piece.piece_type == chess.PAWN
                    and chess.square_rank(sq) in (0, 7)):
                move = chess.Move(self.selected_sq, sq, promotion=chess.QUEEN)

            if move in self.board.legal_moves:
                # ✅ Valid move → make it
                self._player_move(move)
                return

            # Clicked a different friendly piece → re-select
            if piece and piece.color == chess.WHITE:
                self._select(sq)
                return

            # Clicked elsewhere → deselect
            self._deselect()
        else:
            # ── No piece selected yet ─────────────────────────
            if piece and piece.color == chess.WHITE:
                self._select(sq)

    def _select(self, sq):
        """Select a friendly piece and show its legal moves."""
        self.selected_sq    = sq
        self.legal_targets  = [
            m.to_square for m in self.board.legal_moves if m.from_square == sq
        ]
        self._draw_board()

    def _deselect(self):
        """Clear selection."""
        self.selected_sq   = None
        self.legal_targets = []
        self._draw_board()

    def _player_move(self, move):
        """Execute the human player's move."""
        san = self.board.san(move)
        self.board.push(move)
        self.last_move = move
        self._deselect()

        move_num = (len(self.board.move_stack) + 1) // 2
        self._log_move(f"{move_num}. {san:<8}")
        self.move_num_var.set(f"Move:  {move_num}")

        self._draw_board()

        if self._check_game_over():
            return

        self.status_var.set("🤖  AI is thinking...")
        self.root.update()
        self.root.after(150, self._ai_move)   # Short delay so UI updates

    def _ai_move(self):
        """Execute the AI's move."""
        if self.board.is_game_over():
            return

        move = get_best_move(self.board, self.ai_depth)
        if move:
            san = self.board.san(move)
            self.board.push(move)
            self.last_move = move
            self._log_move(f"     {san}\n")

        self._draw_board()

        if not self._check_game_over():
            self.status_var.set("Your turn!  You are White  ♙")
            move_num = (len(self.board.move_stack) + 1) // 2
            self.move_num_var.set(f"Move:  {move_num}")

    # ----------------------------------------------------------
    # GAME LOGIC HELPERS
    # ----------------------------------------------------------

    def _check_game_over(self):
        """Detect and announce game-over conditions."""
        if self.board.is_checkmate():
            winner = "Black (AI)" if self.board.turn == chess.WHITE else "White (You)"
            msg = f"Checkmate!  {winner} wins! 🎉"
            self.status_var.set(msg)
            messagebox.showinfo("Game Over", msg)
            self.game_over = True
            return True

        if self.board.is_stalemate():
            self.status_var.set("Stalemate — It's a draw! 🤝")
            messagebox.showinfo("Game Over", "Stalemate — It's a draw!")
            self.game_over = True
            return True

        if self.board.is_insufficient_material():
            self.status_var.set("Draw — Insufficient material! 🤝")
            messagebox.showinfo("Game Over", "Draw — Insufficient material!")
            self.game_over = True
            return True

        if self.board.is_check():
            self.status_var.set("⚠️  Check!  Your king is under attack!")

        return False

    def _show_hint(self):
        """Calculate and highlight the best move for the player."""
        if self.game_over or self.board.turn != chess.WHITE:
            return
        self.status_var.set("💡  Calculating hint...")
        self.root.update()
        move = get_best_move(self.board, 2)   # Use depth 2 for quick hints
        if move:
            self.selected_sq   = move.from_square
            self.legal_targets = [move.to_square]
            self._draw_board()
            self.status_var.set(f"💡  Hint: try  {self.board.san(move)}")

    def _undo_move(self):
        """Undo the last pair of moves (player + AI)."""
        if len(self.board.move_stack) >= 2:
            self.board.pop()   # Undo AI's move
            self.board.pop()   # Undo player's move
            self.last_move = self.board.peek() if self.board.move_stack else None
            self.game_over = False
            self._deselect()
            self.status_var.set("Move undone!  Your turn. ♙")
            self._log_move("[undo]\n")
        else:
            messagebox.showinfo("Undo", "Nothing to undo!")

    def _new_game(self):
        """Reset everything and start fresh."""
        self.board         = chess.Board()
        self.selected_sq   = None
        self.legal_targets = []
        self.last_move     = None
        self.game_over     = False
        self.move_history  = []
        self.status_var.set("New game!  Your turn.  You are White  ♙")
        self.move_num_var.set("Move:  1")
        self._clear_history()
        self._draw_board()

    def _log_move(self, text):
        """Append a move to the history box."""
        self.history_box.config(state=tk.NORMAL)
        self.history_box.insert(tk.END, text)
        self.history_box.see(tk.END)
        self.history_box.config(state=tk.DISABLED)

    def _clear_history(self):
        self.history_box.config(state=tk.NORMAL)
        self.history_box.delete("1.0", tk.END)
        self.history_box.config(state=tk.DISABLED)


# =============================================================
# PART 5 — MAIN ENTRY POINT
# =============================================================
# This block runs when you execute:  python chess_bot.py

if __name__ == "__main__":
    root = tk.Tk()
    app  = ChessBotApp(root)
    root.mainloop()   # Keeps the window open until you close it
