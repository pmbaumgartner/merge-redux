from typing import Dict, List
from core import LexemeData, WinnerInfo, LineIndex, Lexeme


def get_lines_with_winner(
    winner_info: WinnerInfo, lexeme_data: LexemeData
) -> Dict[LineIndex, List[Lexeme]]:
    result: Dict[LineIndex, List[Lexeme]] = {}
    for line, lexemes in lexeme_data.locations_to_lexemes.items():
        if winner_info in lexemes.values():
            result[line] = [
                lexeme for _, lexeme in sorted(lexemes.items(), key=lambda x: x[0])
            ]
    return result
