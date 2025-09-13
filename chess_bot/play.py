import chess
import chess.pgn
import numpy as np
import bz2
import os
import json
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

def create_model(input_shape, num_classes):
    """
    Creates a simple feedforward neural network for the chess bot.
    """
    input_layer = Input(shape=(input_shape,))
    hidden_layer_1 = Dense(1024, activation='relu')(input_layer)
    hidden_layer_2 = Dense(1024, activation='relu')(hidden_layer_1)
    output_layer = Dense(num_classes, activation='softmax')(hidden_layer_2)

    model = Model(inputs=input_layer, outputs=output_layer)

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model

def board_to_vector(board):
    """
    Converts a chess board to a 773-element vector.
    """
    piece_to_channel = {
        'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
        'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11
    }
    board_vector = np.zeros((8, 8, 12), dtype=np.int8)
    for i in range(8):
        for j in range(8):
            piece = board.piece_at(chess.square(i, j))
            if piece:
                board_vector[i, j, piece_to_channel[piece.symbol()]] = 1

    board_vector = board_vector.flatten()

    castling_rights = np.zeros(4, dtype=np.int8)
    if board.has_kingside_castling_rights(chess.WHITE):
        castling_rights[0] = 1
    if board.has_queenside_castling_rights(chess.WHITE):
        castling_rights[1] = 1
    if board.has_kingside_castling_rights(chess.BLACK):
        castling_rights[2] = 1
    if board.has_queenside_castling_rights(chess.BLACK):
        castling_rights[3] = 1

    side_to_move = np.array([1 if board.turn == chess.WHITE else 0], dtype=np.int8)

    return np.concatenate((board_vector, castling_rights, side_to_move))

def process_and_train(pgn_file):
    """
    Processes a PGN file and trains a model on the data.
    """
    print("Processing data and training model...")
    X = []
    y_uci = []
    with bz2.open(pgn_file, "rt") as pgn:
        game_count = 0
        while True:
            game = chess.pgn.read_game(pgn)
            if game is None:
                break

            board = game.board()
            for move in game.mainline_moves():
                X.append(board_to_vector(board))
                y_uci.append(move.uci())
                board.push(move)
            game_count += 1
            if game_count % 10 == 0:
                print(f"Processed {game_count} games.")

    X = np.array(X)

    unique_moves = sorted(list(set(y_uci)))
    move_to_int = {move: i for i, move in enumerate(unique_moves)}

    y = np.array([move_to_int[move] for move in y_uci])
    y_categorical = to_categorical(y, num_classes=len(unique_moves))

    input_shape = X.shape[1]
    num_classes = len(unique_moves)
    model = create_model(input_shape, num_classes)

    model.fit(X, y_categorical, epochs=10, batch_size=256, validation_split=0.2, verbose=0)

    print("Training complete.")
    return model, move_to_int

def predict_move(board, model, move_to_int):
    """
    Predicts the next move for a given board state.
    """
    legal_moves = [move.uci() for move in board.legal_moves]
    if not legal_moves:
        return None

    board_vector = board_to_vector(board)
    board_vector = np.expand_dims(board_vector, axis=0)

    move_probs = model.predict(board_vector, verbose=0)[0]

    best_move = None
    max_prob = -1
    for move_uci in legal_moves:
        if move_uci in move_to_int:
            move_int = move_to_int[move_uci]
            if move_probs[move_int] > max_prob:
                max_prob = move_probs[move_int]
                best_move = chess.Move.from_uci(move_uci)

    if best_move is None:
        return chess.Move.from_uci(np.random.choice(legal_moves))

    return best_move

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pgn_file = os.path.join(script_dir, "data/ficsgamesdb_pre199911.pgn")
    pgn_file = os.path.normpath(pgn_file)

    model, move_to_int = process_and_train(pgn_file)

    board = chess.Board()
    while not board.is_game_over():
        print("\n" + str(board))
        if board.turn == chess.WHITE:
            # Human player's turn
            move_uci = input("Enter your move (in UCI format, e.g., e2e4): ")
            try:
                move = chess.Move.from_uci(move_uci)
                if move in board.legal_moves:
                    board.push(move)
                else:
                    print("Invalid move. Try again.")
            except:
                print("Invalid move format. Try again.")
        else:
            # Bot's turn
            print("Bot is thinking...")
            bot_move = predict_move(board, model, move_to_int)
            if bot_move:
                print(f"Bot plays: {bot_move.uci()}")
                board.push(bot_move)
            else:
                print("Bot has no legal moves.")
                break

    print("Game over.")
    print(f"Result: {board.result()}")

if __name__ == "__main__":
    main()
