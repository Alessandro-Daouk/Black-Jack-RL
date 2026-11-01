// Sabot fini : un jeu de 52 cartes, rebattu avant une donne s'il en reste moins de 20.
// Même déroulé que src/shoe/game.py : joueur, carte visible, joueur, carte cachée, puis tirages.
// Portée propre : dans le navigateur, seul l'objet `shoe` devient global.
(function () {
"use strict";

const base = typeof module !== "undefined" ? require("./game.js") : game;
const DECK = [4, 4, 4, 4, 4, 4, 4, 4, 4, 16];       // effectifs des valeurs 1..10 dans un jeu
const HILO = [-1, 1, 1, 1, 1, 1, 0, 0, 0, -1];
const RESHUFFLE = 20;

function freshDeck(random) {
  const cards = [];
  base.RANKS.forEach((rank, i) => base.SUITS.forEach((suit) => cards.push({ rank, suit, value: Math.min(i + 1, 10) })));
  for (let i = cards.length - 1; i > 0; i--) {  // Fisher-Yates
    const j = Math.floor(random() * (i + 1));
    [cards[i], cards[j]] = [cards[j], cards[i]];
  }
  return cards;
}

function newShoe(random = Math.random) {
  return { cards: freshDeck(random), pos: 0, unseen: DECK.slice(), full: DECK.slice(), random };
}

// Les cartes ne sont jamais modifiées : une copie partage le tableau, pas les compteurs.
function copy(s) { return { ...s, unseen: s.unseen.slice(), full: s.full.slice() }; }

function draw(s, seen = true) {
  if (s.pos >= s.cards.length) {  // exceptionnel : une donne épuise le sabot, on ajoute un jeu
    s.cards = s.cards.concat(freshDeck(s.random));
    s.unseen = s.unseen.map((k, i) => k + DECK[i]);
    s.full = s.full.map((k, i) => k + DECK[i]);
  }
  const card = s.cards[s.pos++];
  if (seen) s.unseen[card.value - 1] -= 1;
  return card;
}

function remaining(s) { return s.unseen.reduce((a, b) => a + b, 0); }

function trueCount(s) {
  const running = HILO.reduce((sum, h, i) => sum + h * (s.full[i] - s.unseen[i]), 0);
  return running / (remaining(s) / 52);
}

function dealFrom(snapshot) {
  const s = copy(snapshot);
  const p1 = draw(s), up = draw(s), p2 = draw(s), hole = draw(s, false);
  return { s, player: [p1, p2], dealer: [up, hole], actions: [], done: false, gain: null };
}

// Une donne commence par un éventuel rebattage ; l'instantané permet aux modèles de la rejouer.
function startRound(shoe) {
  const reshuffled = shoe.cards.length - shoe.pos < RESHUFFLE;
  const snapshot = reshuffled ? newShoe(shoe.random) : copy(shoe);
  return { snapshot, round: dealFrom(snapshot), reshuffled };
}

function observe(round) {
  const { total, usable } = base.handValue(round.player);
  return { total, upcard: round.dealer[0].value, usable: usable ? 1 : 0, unseen: round.s.unseen.slice() };
}

function hit(round) {
  round.player.push(draw(round.s));
  round.actions.push("HIT");
  if (base.handValue(round.player).total > 21) round.done = true;
}

function stand(round) { round.actions.push("STAND"); round.done = true; }

function finish(round) {
  round.s.unseen[round.dealer[1].value - 1] -= 1;  // la carte cachée est révélée
  const player = base.handValue(round.player).total;
  if (player <= 21) while (base.handValue(round.dealer).total < 17) round.dealer.push(draw(round.s));
  const dealer = base.handValue(round.dealer).total;
  round.gain = player > 21 ? -1 : dealer > 21 || player > dealer ? 1 : player === dealer ? 0 : -1;
  return round;
}

// --- Modèles : agents linéaires dans la composition, régression logistique, ou table du paquet infini ---

const dot = (w, x) => w.reduce((sum, v, i) => sum + v * x[i], 0);

function features(unseen, data) {
  const n = unseen.reduce((a, b) => a + b, 0);
  const z = unseen.map((k) => k / n).concat([n / 52]);
  return [1].concat(z.map((v, i) => (v - data.mu[i]) / data.sd[i]));
}

function logisticStand(total, upcard, unseen, data) {
  const n = unseen.reduce((a, b) => a + b, 0);
  const z = [1].concat(unseen.map((k) => k / n), [n / 52]);
  const { W, b } = data.logistic;
  const logits = W.map((row, c) => dot(row[upcard - 1], z) + b[c]);
  const top = Math.max(...logits);
  const p = logits.map((l) => Math.exp(l - top));
  const sum = p.reduce((a, v) => a + v, 0);
  // issues 17, 18, 19, 20, 21, BUST : gain +1 si le croupier saute ou fait moins, 0 à égalité, −1 sinon
  return p.reduce((acc, v, k) => acc + (v / sum) * (k === 5 ? 1 : Math.sign(total - 17 - k)), 0);
}

function advice(model, obs, data) {
  const { total, upcard, usable, unseen } = obs;
  if (model.kind === "table") return base.advice(model, obs);
  const f = features(unseen, data);
  const cell = (a) => model.q[a][total][upcard][usable];
  const hitValue = dot(cell(1), f);
  const standValue = model.standFrom === "logistique" ? logisticStand(total, upcard, unseen, data) : dot(cell(0), f);
  return { action: hitValue > standValue ? "HIT" : "STAND", stand: standValue, hit: hitValue };
}

function playModel(model, snapshot, data) {
  const round = dealFrom(snapshot);
  while (!round.done) {
    if (advice(model, observe(round), data).action === "HIT") hit(round); else stand(round);
  }
  return finish(round);
}

const shoe = { DECK, RESHUFFLE, newShoe, copy, draw, remaining, trueCount, startRound, dealFrom, observe, hit, stand,
               finish, features, logisticStand, advice, playModel };
if (typeof module !== "undefined") module.exports = shoe; else window.shoe = shoe;
})();
