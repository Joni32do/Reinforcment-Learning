import random

class Card:
    def __init__(self, rank, suit):
        self.rank = rank
        self.suit = suit

    def __str__(self):
        return f"{self.rank} of {self.suit}"

class Deck:
    def __init__(self):
        self.cards = []

    def add_card(self, card):
        self.cards.append(card)

    def remove_card(self, card):
        self.cards.remove(card)

    def shuffle(self):
        random.shuffle(self.cards)

    def deal(self, num_cards):
        if num_cards > len(self.cards):
            raise ValueError("Not enough cards in the deck.")
        return [self.cards.pop() for _ in range(num_cards)]
        


class PokerDeck(Deck):
    def __init__(self):
        super().__init__()
        for suit in ["CLUB", "SPADES", "HEART", "DIAMOND"]:
            for rank in ["2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "D", "K", "A"]:
                self.add_card(Card(rank, suit))


class DoppelkopfDeck(Deck):
    def __init__(self):
        super().__init__()
        for suit in ["CLUB", "SPADES", "HEART", "DIAMOND"]:
            for rank in ["9", "10", "J", "D", "K", "A"]:
                for _ in range(2):
                    self.add_card(Card(rank, suit))


class SkatDeck(Deck):
    def __init__(self):
        super().__init__()
        for suit in ["CLUB", "SPADES", "HEART", "DIAMOND"]:
            for rank in ["7", "8", "9", "10", "J", "D", "K", "A"]:
                self.add_card(Card(rank, suit))
            


    def __init__(self):
        super().__init__()
        for suit in ["CLUB", "SPADES", "HEART", "DIAMOND"]:
            for rank in ["2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "D", "K", "A"]:
                self.add_card(Card(rank, suit))