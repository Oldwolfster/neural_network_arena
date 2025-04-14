import random
from collections import Counter
from itertools import combinations
import eval7  # fast poker hand evaluator

# Convert human-readable hands to eval7 format
hand1 = [eval7.Card('Qh'), eval7.Card('2d')]  # Q2 offsuit
hand2 = [eval7.Card('Tc'), eval7.Card('4c')]  # T4 suited

# Run Monte Carlo simulation
iterations = 100000
wins = [0, 0, 0]  # [hand1 win, hand2 win, tie]

deck = eval7.Deck()
for _ in range(iterations):
    deck.shuffle()
    # Remove hand cards from deck
    for card in hand1 + hand2:
        deck.cards.remove(card)

    # Deal 5 community cards
    board = deck.peek(5)

    score1 = eval7.evaluate(hand1 + board)
    score2 = eval7.evaluate(hand2 + board)

    if score1 > score2:
        wins[0] += 1
    elif score2 > score1:
        wins[1] += 1
    else:
        wins[2] += 1

    deck = eval7.Deck()  # reset the deck

# Convert results to percentages
total = sum(wins)
odds = {
    'Q2 offsuit win %': wins[0] / total * 100,
    'T4 suited win %': wins[1] / total * 100,
    'Tie %': wins[2] / total * 100
}

