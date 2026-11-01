// node --test web/shoe.test.js
"use strict";
const test = require("node:test");
const assert = require("node:assert");
const game = require("./game.js");
const shoe = require("./shoe.js");

global.window = {};
require("./models_shoe.js");
const data = window.SHOE;

function seeded(seed) {  // mulberry32
  return () => {
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

test("les agents donnent en JavaScript les mêmes valeurs qu'en Python", () => {
  for (const check of data.checks) {
    for (const [id, [stand, hit]] of Object.entries(check.values)) {
      const model = data.models.find((m) => m.id === id);
      const a = shoe.advice(model, check, data);
      assert.ok(Math.abs(a.stand - stand) < 1e-9 && Math.abs(a.hit - hit) < 1e-9, `${id} ${JSON.stringify(check)}`);
    }
  }
});

test("les cartes non vues restent cohérentes et le sabot est rebattu sous 20 cartes", () => {
  const random = seeded(3);
  let current = shoe.newShoe(random), reshuffles = 0;
  for (let i = 0; i < 2000; i++) {
    const { snapshot, round, reshuffled } = shoe.startRound(current);
    if (reshuffled) { reshuffles += 1; assert.strictEqual(snapshot.cards.length, 52); }
    while (!round.done) { if (random() < 0.5) shoe.hit(round); else shoe.stand(round); }
    shoe.finish(round);
    current = round.s;
    const counts = Array(10).fill(0);
    current.cards.slice(current.pos).forEach((c) => { counts[c.value - 1] += 1; });
    assert.deepStrictEqual(current.unseen, counts);
  }
  assert.ok(reshuffles > 200);
});

test("chaque modèle rejoue la même donne et retrouve son gain estimé", () => {
  const random = seeded(11);
  for (const model of data.models) {
    let gains = 0, rounds = 0, current = shoe.newShoe(random);
    for (let i = 0; i < 120000; i++) {
      const { snapshot } = shoe.startRound(current);
      const round = shoe.playModel(model, snapshot, data);
      gains += round.gain; rounds += 1; current = round.s;
    }
    // deux estimations Monte-Carlo indépendantes : tolérance de 4 erreurs-types de leur différence
    const tolerance = 4 * 0.95 * Math.sqrt(1 / rounds + 1 / data.evalRounds);
    assert.ok(Math.abs(gains / rounds - model.estimatedGain) < tolerance, `${model.id} : ${gains / rounds}`);
  }
});

test("une copie du sabot ne modifie pas l'original", () => {
  const s = shoe.newShoe(seeded(5));
  const before = s.unseen.slice();
  const round = shoe.dealFrom(s);
  shoe.hit(round);
  assert.deepStrictEqual(s.unseen, before);
  assert.strictEqual(s.pos, 0);
  assert.strictEqual(game.handValue(round.player).total > 0, true);
});
