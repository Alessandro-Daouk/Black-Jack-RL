// node --test web/game.test.js
"use strict";
const test = require("node:test");
const assert = require("node:assert");
const game = require("./game.js");

global.window = {};
require("./models.js");
const { models } = window.MODELS;

const card = (value) => ({ rank: String(value), suit: "♠", value });

function seeded(seed) {  // mulberry32 : simulations reproductibles
  return () => {
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

test("valeur des mains et as utilisable", () => {
  assert.deepStrictEqual(game.handValue([card(1), card(10)]), { total: 21, usable: true });
  assert.deepStrictEqual(game.handValue([card(1), card(1), card(9)]), { total: 21, usable: true });
  assert.deepStrictEqual(game.handValue([card(1), card(9), card(5)]), { total: 15, usable: false });
  assert.strictEqual(game.handValue([card(10), card(10), card(5)]).total, 25);
});

test("le croupier reste sur 17 souple et les gains suivent les règles", () => {
  const round = { dealer: [card(1), card(6)], dealerCard() { throw new Error("ne doit pas tirer"); } };
  game.playDealer(round);
  assert.strictEqual(game.outcome({ cards: [card(10), card(8)] }, round), 1);
  assert.strictEqual(game.outcome({ cards: [card(10), card(7)] }, round), 0);
  assert.strictEqual(game.outcome({ cards: [card(10), card(10), card(5)] }, { dealer: [card(10), card(10), card(5)] }), -1);
});

test("tous les sièges reçoivent les mêmes cartes", () => {
  const round = game.newRound(seeded(1));
  const a = game.newSeat(), b = game.newSeat();
  game.deal(round, a); game.deal(round, b);
  game.hit(round, a); game.hit(round, b);
  assert.deepStrictEqual(a.cards, b.cards);
});

test("le gain simulé de chaque modèle retrouve son gain exact", () => {
  const random = seeded(2026), n = 200000;
  for (const model of models) {
    let total = 0;
    for (let i = 0; i < n; i++) {
      const round = game.newRound(random), seat = game.newSeat();
      game.playModel(round, seat, model);
      game.playDealer(round);
      total += game.outcome(seat, round);
    }
    // écart-type d'une partie ≈ 0,95 : tolérance de 4 erreurs-types
    assert.ok(Math.abs(total / n - model.exactGain) < 4 * 0.95 / Math.sqrt(n), `${model.id} : ${total / n}`);
  }
});
