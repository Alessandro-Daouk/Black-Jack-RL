// Interface : une donne, vos décisions, puis la même donne jouée par chaque modèle.
// Deux modes, paquet infini et sabot fini, derrière les mêmes méthodes.
"use strict";

const $ = (id) => document.getElementById(id);
const fmt = (x, digits = 3) =>
  x.toLocaleString("fr-FR", { minimumFractionDigits: digits, maximumFractionDigits: digits, signDisplay: "exceptZero" })
    .replace("-", "−");
const label = (action) => (action === "HIT" ? "Tirer" : "Rester");
const newScores = (models) => ({ you: { sum: 0, n: 0 }, ...Object.fromEntries(models.map((m) => [m.id, { sum: 0, n: 0 }])) });

const infinite = {
  models: window.MODELS.models, reference: "optimal", referenceName: "la stratégie optimale",
  gainKey: "exactGain", gainHead: "Exact", gainNote: "Le gain exact attendu est calculé sans simulation, par l'oracle du projet.",
  start() { this.round = game.newRound(); this.seat = game.newSeat(); game.deal(this.round, this.seat); },
  player() { return this.seat.cards; },
  dealer() { return this.round.dealer; },
  observe() { return game.observation(this.seat, this.round); },
  advice: (model, obs) => game.advice(model, obs),
  hit() { game.hit(this.round, this.seat); },
  stand() { game.stand(this.seat); },
  done() { return this.seat.done; },
  finish() {
    const seats = Object.fromEntries(this.models.map((m) => {
      const seat = game.newSeat();
      game.playModel(this.round, seat, m);
      return [m.id, seat];
    }));
    game.playDealer(this.round);
    for (const seat of Object.values(seats)) { seat.gain = game.outcome(seat, this.round); seat.dealer = this.round.dealer; }
    return { gain: game.outcome(this.seat, this.round), seats };
  },
  info: () => "",
};

const finite = window.SHOE && {
  models: window.SHOE.models, reference: "hybride", referenceName: "l'hybride linéaire, le meilleur modèle mesuré",
  gainKey: "estimatedGain", gainHead: "Estimé",
  gainNote: `Gain estimé sur ${window.SHOE.evalRounds.toLocaleString("fr-FR")} manches simulées : pas d'oracle exact en sabot fini.`,
  sabot: null,
  start() {
    const { snapshot, round, reshuffled } = shoe.startRound(this.sabot || shoe.newShoe());
    Object.assign(this, { snapshot, round, reshuffled });
  },
  player() { return this.round.player; },
  dealer() { return this.round.dealer; },
  observe() { return shoe.observe(this.round); },
  advice: (model, obs) => shoe.advice(model, obs, window.SHOE),
  hit() { shoe.hit(this.round); },
  stand() { shoe.stand(this.round); },
  done() { return this.round.done; },
  finish() {
    // chaque modèle rejoue la donne depuis le même sabot, dans le même ordre
    const seats = Object.fromEntries(this.models.map((m) => {
      const r = shoe.playModel(m, this.snapshot, window.SHOE);
      return [m.id, { cards: r.player, actions: r.actions, gain: r.gain, dealer: r.dealer }];
    }));
    shoe.finish(this.round);
    this.sabot = this.round.s;  // le sabot continue avec vos cartes
    return { gain: this.round.gain, seats };
  },
  info() {
    const s = this.round ? this.round.s : this.sabot;
    if (!s) return "Sabot neuf de 52 cartes.";
    return `${this.reshuffled && phase === "playing" ? "Sabot rebattu · " : ""}${shoe.remaining(s)} cartes non vues · `
      + `compte vrai Hi-Lo ${fmt(shoe.trueCount(s), 1)}`;
  },
};

const modes = { infini: infinite, ...(finite ? { sabot: finite } : {}) };
let mode = infinite, phase = "idle", conform = [], results = {};
for (const m of Object.values(modes)) m.scores = newScores(m.models);

function cardElement(card, { small = false, hidden = false, flip = false } = {}) {
  const el = document.createElement("div");
  el.className = `card${small ? " small" : ""}${hidden ? " back" : ""}${flip ? " flip" : ""}`;
  if (hidden) { el.setAttribute("aria-label", "carte cachée"); return el; }
  if (card.suit === "♥" || card.suit === "♦") el.classList.add("red");
  el.setAttribute("aria-label", `${card.rank} ${card.suit}`);
  el.innerHTML = `<span class="corner">${card.rank}<br>${card.suit}</span><span class="pip">${card.suit}</span>`;
  return el;
}

// N'ajoute que les nouvelles cartes, pour que seules elles soient animées.
function showCards(container, cards, options = {}) {
  cards.slice(container.children.length).forEach((c) => container.append(cardElement(c, options)));
}

function showTotal(el, cards) {
  const { total, usable } = game.handValue(cards);
  el.textContent = total > 21 ? `${total} · dépassé` : usable ? `${total} (as = 11)` : String(total);
  el.classList.toggle("bust", total > 21);
}

function renderTable() {
  const dealer = $("dealer-cards");
  if (phase === "playing") {
    const up = mode.dealer()[0];
    $("dealer-total").textContent = `${up.rank === "A" ? "As" : up.value} visible`;
  } else if (phase === "done") {
    if (dealer.children[1].classList.contains("back")) dealer.children[1].replaceWith(cardElement(mode.dealer()[1], { flip: true }));
    showCards(dealer, mode.dealer());
    showTotal($("dealer-total"), mode.dealer());
  }
  if (phase !== "idle") { showCards($("player-cards"), mode.player()); showTotal($("player-total"), mode.player()); }
  $("hit").disabled = $("stand").disabled = $("hint").disabled = phase !== "playing";
  $("deal").disabled = phase === "playing";
  $("shoe-info").textContent = mode.info();
}

function renderSeats() {
  $("seats").replaceChildren(...mode.models.map((model) => {
    const seat = results[model.id];
    const panel = document.createElement("article");
    panel.className = "seat";
    const badge = !seat ? "" : `<span class="badge ${seat.gain > 0 ? "win" : seat.gain < 0 ? "loss" : "push"}">${
      seat.gain > 0 ? "+1 · gagné" : seat.gain < 0 ? "−1 · perdu" : "0 · égalité"}</span>`;
    panel.innerHTML = `<div class="seat-head"><span class="seat-name">${model.name}</span>${badge}</div>
      <p class="seat-detail" title="${model.detail}">${model.detail}</p><div class="cards"></div><div class="trail"></div>`;
    const cards = panel.querySelector(".cards"), trail = panel.querySelector(".trail");
    if (seat) {
      showCards(cards, seat.cards, { small: true });
      const total = game.handValue(seat.cards).total, dealer = game.handValue(seat.dealer).total;
      trail.innerHTML = seat.actions.map((a) => `<span class="chip">${label(a)}</span>`).join("")
        + `<span class="chip">${total > 21 ? `${total}, dépassé` : `total ${total}`}</span>`
        + (mode === finite ? `<span class="chip">croupier ${dealer > 21 ? `${dealer}, dépassé` : dealer}</span>` : "");
    } else if (phase === "playing") {
      cards.append(cardElement(null, { small: true, hidden: true }), cardElement(null, { small: true, hidden: true }));
      trail.innerHTML = `<span class="waiting">Joue la même donne après vous.</span>`;
    } else {
      trail.innerHTML = `<span class="waiting">En attente d'une donne.</span>`;
    }
    return panel;
  }));
}

function renderScores() {
  const { scores } = mode;
  const rows = [{ id: "you", name: "Vous", ref: null }, ...mode.models.map((m) => ({ id: m.id, name: m.name, ref: m[mode.gainKey] }))];
  const best = Math.max(...rows.map((r) => scores[r.id].sum));
  $("gain-head").textContent = mode.gainHead;
  $("score-rows").replaceChildren(...rows.map((r) => {
    const { sum, n } = scores[r.id];
    const tr = document.createElement("tr");
    if (scores.you.n && sum === best && rows.some((o) => scores[o.id].sum !== best)) tr.className = "leader";
    tr.innerHTML = `<td>${r.name}</td><td class="${sum > 0 ? "positive" : sum < 0 ? "negative" : ""}">${fmt(sum, 0)}</td>
      <td>${n ? fmt(sum / n) : "—"}</td><td>${r.ref === null ? "—" : fmt(r.ref, 4)}</td>`;
    return tr;
  }));
  $("score-note").textContent = scores.you.n
    ? `${scores.you.n} partie${scores.you.n > 1 ? "s" : ""} : sur si peu, la chance domine (écart-type 0,95 par partie). ${mode.gainNote}`
    : mode.gainNote;
}

function renderHint() {
  const panel = $("hint-panel");
  if (panel.hidden || phase !== "playing") return;
  const obs = mode.observe();
  panel.replaceChildren(...mode.models.map((model) => {
    const a = mode.advice(model, obs);
    const row = document.createElement("div");
    row.className = "hint-row";
    row.innerHTML = `<span><strong>${model.name}</strong> · ${label(a.action)}</span>
      <span class="values">STAND ${fmt(a.stand)} · HIT ${fmt(a.hit)}</span>`;
    return row;
  }));
}

function renderCount() {
  const n = mode.scores.you.n;
  $("round-count").textContent = phase === "playing" ? `Donne ${n + 1}` : n ? `${n} donne${n > 1 ? "s" : ""} jouée${n > 1 ? "s" : ""}` : "Aucune donne";
}

function setStatus(html) { $("status").innerHTML = html; }

function render() { renderTable(); renderSeats(); renderScores(); renderHint(); renderCount(); }

function clearTable() {
  ["dealer-cards", "player-cards"].forEach((id) => $(id).replaceChildren());
  ["dealer-total", "player-total"].forEach((id) => { $(id).textContent = ""; $(id).classList.remove("bust"); });
}

function startRound() {
  mode.start();
  phase = "playing"; conform = []; results = {};
  clearTable();
  $("dealer-cards").append(cardElement(mode.dealer()[0]), cardElement(null, { hidden: true }));
  if (game.handValue(mode.player()).total === 21) { mode.stand(); return finish(); }
  render();
  const obs = mode.observe(), up = mode.dealer()[0];
  setStatus(`Vous avez <strong>${obs.total}</strong> face à ${up.rank === "A" ? "un <strong>as</strong>" : `un <strong>${up.value}</strong>`}. Tirer ou rester ?`);
}

function act(action) {
  if (phase !== "playing") return;
  const reference = mode.models.find((m) => m.id === mode.reference);
  conform.push(mode.advice(reference, mode.observe()).action === action);
  if (action === "HIT") mode.hit(); else mode.stand();
  if (!mode.done() && game.handValue(mode.player()).total === 21) mode.stand();  // à 21, tirer ne peut que perdre
  if (mode.done()) return finish();
  render();
  setStatus(`Vous avez <strong>${game.handValue(mode.player()).total}</strong>. Tirer ou rester ?`);
}

function finish() {
  phase = "done";
  const { gain, seats } = mode.finish();
  results = seats;
  const { scores } = mode;
  scores.you.sum += gain; scores.you.n += 1;
  for (const [id, seat] of Object.entries(seats)) { scores[id].sum += seat.gain; scores[id].n += 1; }
  $("hint-panel").hidden = true; $("hint").setAttribute("aria-expanded", "false");
  render();
  const verdict = gain > 0 ? "victoire (+1)" : gain < 0 ? "défaite (−1)" : "égalité (0)";
  const dealer = game.handValue(mode.dealer()).total;
  const good = conform.filter(Boolean).length;
  setStatus(`<strong>${verdict}</strong> · croupier ${dealer > 21 ? `${dealer}, dépassé` : dealer}`
    + (conform.length ? ` · ${good} décision${good > 1 ? "s" : ""} sur ${conform.length} conforme${conform.length > 1 ? "s" : ""} à ${mode.referenceName}.` : " · 21 servi."));
}

function toggleHint() {
  if (phase !== "playing") return;
  const panel = $("hint-panel");
  panel.hidden = !panel.hidden;
  $("hint").setAttribute("aria-expanded", String(!panel.hidden));
  renderHint();
}

function switchMode(key) {
  if (!modes[key] || modes[key] === mode) return;
  mode = modes[key];
  phase = "idle"; results = {};
  clearTable();
  $("hint-panel").hidden = true;
  document.querySelectorAll("[data-mode]").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.mode === key)));
  render();
  setStatus(key === "sabot"
    ? "Sabot de 52 cartes : les cartes déjà sorties changent les chances, et le sabot continue d'une donne à l'autre."
    : "Paquet infini : chaque carte est tirée au hasard. Appuyez sur Espace pour distribuer.");
}

$("deal").addEventListener("click", startRound);
$("hit").addEventListener("click", () => act("HIT"));
$("stand").addEventListener("click", () => act("STAND"));
$("hint").addEventListener("click", toggleHint);
$("open-rules").addEventListener("click", () => $("rules").showModal());
document.querySelectorAll("[data-mode]").forEach((b) => b.addEventListener("click", () => switchMode(b.dataset.mode)));
if (!finite) document.querySelector(".segmented").hidden = true;
$("reset").addEventListener("click", () => {
  Object.values(mode.scores).forEach((s) => { s.sum = 0; s.n = 0; });
  renderScores(); renderCount();
});
document.addEventListener("keydown", (event) => {
  if ($("rules").open || event.metaKey || event.ctrlKey || event.altKey) return;
  const key = event.key.toLowerCase();
  if (key === "h") act("HIT");
  else if (key === "s") act("STAND");
  else if (key === "c") toggleHint();
  else if ((key === " " || key === "n") && phase !== "playing") { event.preventDefault(); startRound(); }
});

render();
