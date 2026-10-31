// Logique du jeu, sans interface : règles de Blackjack-v1 (Gymnasium), celles de l'entraînement.
// Paquet infini, croupier qui reste sur 17 (même souple), blackjack payé 1:1, ni double ni partage.
"use strict";

const SUITS = ["♠", "♥", "♦", "♣"];
const RANKS = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"];

function drawCard(random = Math.random) {
  const rank = Math.floor(random() * 13);  // 13 rangs équiprobables : un 10 vaut 4/13, comme dans Gymnasium
  return { rank: RANKS[rank], suit: SUITS[Math.floor(random() * 4)], value: Math.min(rank + 1, 10) };
}

function handValue(cards) {
  const hard = cards.reduce((sum, c) => sum + c.value, 0);
  const usable = cards.some((c) => c.value === 1) && hard + 10 <= 21;
  return { total: usable ? hard + 10 : hard, usable };
}

// Une donne : tous les sièges reçoivent les mêmes deux cartes, puis la k-ième carte tirée par
// chaque siège est la même ; le croupier tire dans son propre flux. Le duel est donc équitable.
function newRound(random = Math.random) {
  const stream = [drawCard(random), drawCard(random)];
  const dealer = [drawCard(random), drawCard(random)];
  return {
    random, stream, dealer,
    card(k) { while (this.stream.length <= k) this.stream.push(drawCard(this.random)); return this.stream[k]; },
    dealerCard() { const c = drawCard(this.random); this.dealer.push(c); return c; },
  };
}

function newSeat() { return { cards: null, actions: [], done: false }; }

function deal(round, seat) { seat.cards = [round.card(0), round.card(1)]; seat.actions = []; seat.done = false; }

function hit(round, seat) {
  seat.cards.push(round.card(seat.cards.length));
  seat.actions.push("HIT");
  if (handValue(seat.cards).total > 21) seat.done = true;
}

function stand(seat) { seat.actions.push("STAND"); seat.done = true; }

function observation(seat, round) {
  const { total, usable } = handValue(seat.cards);
  return { total, upcard: round.dealer[0].value, usable: usable ? 1 : 0 };
}

function advice(model, obs) {
  const { total, upcard, usable } = obs;
  return { action: model.policy[total][upcard][usable] ? "HIT" : "STAND",
           stand: model.stand[total][upcard][usable], hit: model.hit[total][upcard][usable] };
}

function playModel(round, seat, model) {
  deal(round, seat);
  while (!seat.done) {
    if (advice(model, observation(seat, round)).action === "HIT") hit(round, seat); else stand(seat);
  }
}

function playDealer(round) {
  while (handValue(round.dealer).total < 17) round.dealerCard();
}

function outcome(seat, round) {
  const player = handValue(seat.cards).total;
  const dealer = handValue(round.dealer).total;
  if (player > 21) return -1;
  if (dealer > 21 || player > dealer) return 1;
  return player === dealer ? 0 : -1;
}

const game = { drawCard, handValue, newRound, newSeat, deal, hit, stand, observation, advice,
               playModel, playDealer, outcome };
if (typeof module !== "undefined") module.exports = game;
