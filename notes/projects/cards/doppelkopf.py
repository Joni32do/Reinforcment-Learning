# This is the direct approach for reinforcement learning in Doppelkopf.
#
# Not so much fancy classes but firstly focusing on the game logic, which
# afterwards can be wrapped inside of nice classes, for gui enabling and stuff

import numpy as np
from typing import Iterable
from itertools import product

class Cards:
    def __init__(self, cards: Iterable[int]):
        self.cards = cards
        

    def __str__(self):
        i = 0
        for suit in ["CLUB", "SPADES", "HEART", "DIAMOND"]:
            for rank in ["A", "10", "K", "Q", "J", "9"]:
                    self.lookup[i] = (rank, suit)
                    i += 1
        suits = {
            "CLUB": "\u2663",
            "SPADES": "\u2660",
            "HEART": "\u2665",
            "DIAMOND": "\u2666"
        }
        output = ""
        for card in self.cards:
            rank, suit = self.lookup[card]
            output += f"[{rank} {suits[suit]}] "
        return output
    

    def dealDeck(self) -> Iterable[Cards]:
        players_cards = [[], [], [], []]
        # It is symmetric to deal 'two' times the half deck
        for _ in range(2):
            deck = np.arange(24)
            np.random.shuffle(deck)
            for i in range(4):
                players_cards[i].append(deck[6*i:6*(i+1)])
        return [Cards(cards) for cards in players_cards]
    
    
    def isTrump(self, card: int) -> bool:
        # isDiamond = (card // 6 == 3)
        # isQueen = (card % 6 == 3)
        # isJack = (card % 6 == 4)
        # isHeart10 = (card == 13)
        # return isQueen or isJack or isDiamond or isHeart10
        return (card // 6 == 3) or (card % 6 == 3) or (card % 6 == 4) or (card == 13)
    

    def getCards(self) -> Iterable[int]:
        return self.cards
    

    def hasTrump(self) -> bool:
        return any([self.isTrump(card) for card in self.cards])
    

    def getTrump(self) -> Iterable[int]:
        return [card for card in self.cards if self.isTrump(card)]
    

    def hasSuit(self, suit: int) -> bool:
        return any([card // 6 == suit for card in self.cards])
    

    def getSuit(self, suit: int) -> Iterable[int]:
        return [card for card in self.cards if card // 6 == suit]


def get_state_space(playedCards: Iterable[Cards],
                    currentPoints: Iterable[int],
                    isRe: bool,
                    ) -> int:
    # TODO:
    # * Ansagen
    # * Solo
    pass


def game_logic():
    pass


def possible_actions(hand: Cards, card_played: int) -> Iterable[int]:
    card_played = card_played % 24 # Debug: Double dont matter
    suit = card_played // 6 # CLUB, SPADES, HEART, DIAMOND
    rank = card_played % 6 # A, 10, K, Q, J, 9
    if card_played == -1:
        return hand.getCards()
    # Bedienpflicht
    if hand.isTrump(card_played):
        if hand.hasTrump():
            return hand.getTrump()
        else:
            return hand.getCards()
    elif hand.hasSuit(suit):
            return hand.getSuit(suit)
    

