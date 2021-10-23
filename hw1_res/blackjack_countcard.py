import gym
from gym import spaces
from gym.utils import seeding


def cmp(a, b):
    return float(a > b) - float(a < b)


# 1 = Ace, 2-10 = Number cards, Jack/Queen/King = 10
deck = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10]


def usable_ace(hand):  # Does this hand have a usable ace?
    return 1 in hand and sum(hand) + 10 <= 21


def sum_hand(hand):  # Return current hand total
    if usable_ace(hand):
        return sum(hand) + 10
    return sum(hand)


def is_bust(hand):  # Is this hand a bust?
    return sum_hand(hand) > 21


def score(hand):  # What is the score of this hand (0 if bust)
    return 0 if is_bust(hand) else sum_hand(hand)


def is_natural(hand):  # Is this hand a natural blackjack?
    return sorted(hand) == [1, 10]


class BlackjackCountCardEnv(gym.Env):
    """Simple blackjack environment

    Blackjack is a card game where the goal is to obtain cards that sum to as
    near as possible to 21 without going over.  They're playing against a fixed
    dealer.
    Face cards (Jack, Queen, King) have point value 10.
    Aces can either count as 11 or 1, and it's called 'usable' at 11.
    This game is placed with an infinite deck (or with replacement).
    The game starts with dealer having one face up and one face down card, while
    player having two face up cards. (Virtually for all Blackjack games today).

    The player can request additional cards (hit=1) until they decide to stop
    (stick=0) or exceed 21 (bust).

    After the player sticks, the dealer reveals their facedown card, and draws
    until their sum is 17 or greater.  If the dealer goes bust the player wins.

    If neither player nor dealer busts, the outcome (win, lose, draw) is
    decided by whose sum is closer to 21.  The reward for winning is +1,
    drawing is 0, and losing is -1.

    The observation of a 3-tuple of: the players current sum,
    the dealer's one showing card (1-10 where 1 is ace),
    and whether or not the player holds a usable ace (0 or 1).

    This environment corresponds to the version of the blackjack problem
    described in Example 5.1 in Reinforcement Learning: An Introduction
    by Sutton and Barto.
    http://incompleteideas.net/book/the-book-2nd.html
    """

    def __init__(self, counting_system, natural=False, sab=False):
        self.action_space = spaces.Discrete(3)

        if counting_system == "plus-minus":
            self.observation_space = spaces.Tuple(
                (spaces.Discrete(32), spaces.Discrete(11), spaces.Discrete(3), spaces.Discrete(42))
            )
            self.counting_system = {
                1: -1,
                2: 1,
                3: 1,
                4: 1,
                5: 1,
                6: 1,
                7: 0,
                8: 0,
                9: 0,
                10: -1,
            }
            # диапазон от -20 до 20, всего 41

        elif counting_system == "halves":
            self.observation_space = spaces.Tuple(
                (spaces.Discrete(32), spaces.Discrete(11), spaces.Discrete(3), spaces.Discrete(90))
            )
            self.counting_system = {
                1: -2,
                2: 1,
                3: 2,
                4: 2,
                5: 3,
                6: 2,
                7: 1,
                8: 0,
                9: -1,
                10: -2,
            }
            # диапазон от -44 до 44, всего 89

        self.seed()

        self.deck_cards = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10] * 4
        self.sum_cards = 0

        self.natural = natural
        self.sab = sab

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def draw_card(self, np_random):
        card = int(np_random.choice(self.deck_cards))
        self.deck_cards.remove(card)
        return card

    def draw_hand(self, np_random):
        return [self.draw_card(np_random), self.draw_card(np_random)]

    def step(self, action):
        assert self.action_space.contains(action)
        if action == 2:  # hit: add a card to players hand and return
            new_card = self.draw_card(self.np_random)
            self.player.append(new_card)
            self.sum_cards += self.counting_system[new_card]
            done = True
            while sum_hand(self.dealer) < 17:
                new_card = self.draw_card(self.np_random)
                self.dealer.append(new_card)
                self.sum_cards += self.counting_system[new_card]
            reward = cmp(score(self.player), score(self.dealer)) * 2
        elif action == 1:  # hit: add a card to players hand and return
            new_card = self.draw_card(self.np_random)
            self.player.append(new_card)
            self.sum_cards += self.counting_system[new_card]
            if is_bust(self.player):
                done = True
                reward = -1.0
            else:
                done = False
                reward = 0.0
        elif action == 0:  # stick: play out the dealers hand, and score
            done = True
            while sum_hand(self.dealer) < 17:
                new_card = self.draw_card(self.np_random)
                self.dealer.append(new_card)
                self.sum_cards += self.counting_system[new_card]

            reward = cmp(score(self.player), score(self.dealer))
            if self.sab and is_natural(self.player) and not is_natural(self.dealer):
                # Player automatically wins. Rules consistent with S&B
                reward = 1.0
            elif (
                not self.sab
                and self.natural
                and is_natural(self.player)
                and reward == 1.0
            ):
                # Natural gives extra points, but doesn't autowin. Legacy implementation
                reward = 1.5
        return self._get_obs(), reward, done, {}

    def _get_obs(self):
        return (sum_hand(self.player), self.dealer[0], usable_ace(self.player), self.sum_cards)

    def reset(self):
        if len(self.deck_cards) < 15.:
            self.deck_cards = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10] * 4
            self.sum_cards = 0
        self.dealer = self.draw_hand(self.np_random)
        self.player = self.draw_hand(self.np_random)
        open_cards = self.dealer[0], self.player[0], self.player[1]
        for card_ in open_cards:
            self.sum_cards += self.counting_system[card_]
        return self._get_obs()
