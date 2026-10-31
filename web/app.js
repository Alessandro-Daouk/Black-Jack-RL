// Interface : une donne, vos décisions, puis la même donne jouée par chaque modèle.
"use strict";

const { models } = window.MODELS;
const optimal = models.find((m) => m.id === "optimal");
const $ = (id) => document.getElementById(id);
const fmt = (x, digits = 3) =>
  x.toLocaleString("fr-FR", { minimumFractionDigits: digits, maximumFractionDigits: digits, signDisplay: "exceptZero" })
    .replace("-", "−");
const label = (action) => (action === "HIT" ? "Tirer" : "Rester");

let round = null, you = null, phase = "idle", conform = [], results = {};
const scores = { you: { sum: 0, n: 0 }, ...Object.fromEntries(models.map((m) => [m.id, { sum: 0, n: 0 }])) };

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
  if (options.reset) container.replaceChildren();
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
    dealer.replaceChildren(cardElement(round.dealer[0]), cardElement(null, { hidden: true }));
    $("dealer-total").textContent = `${round.dealer[0].rank === "A" ? "As" : round.dealer[0].value} visible`;
  } else if (phase === "done") {
    dealer.children[1].replaceWith(cardElement(round.dealer[1], { flip: true }));
    showCards(dealer, round.dealer);
    showTotal($("dealer-total"), round.dealer);
  }
  if (you) { showCards($("player-cards"), you.cards); showTotal($("player-total"), you.cards); }
  $("hit").disabled = $("stand").disabled = $("hint").disabled = phase !== "playing";
  $("deal").disabled = phase === "playing";
}

function renderSeats() {
  $("seats").replaceChildren(...models.map((model) => {
    const seat = results[model.id];
    const panel = document.createElement("article");
    panel.className = "panel seat";
    const badge = !seat ? "" : `<span class="badge ${seat.gain > 0 ? "win" : seat.gain < 0 ? "loss" : "push"}">${
      seat.gain > 0 ? "+1 · gagné" : seat.gain < 0 ? "−1 · perdu" : "0 · égalité"}</span>`;
    panel.innerHTML = `<div class="seat-head"><span class="seat-name">${model.name}</span>${badge}</div>
      <p class="seat-detail" title="${model.detail}">${model.detail}</p><div class="cards"></div><div class="trail"></div>`;
    const cards = panel.querySelector(".cards"), trail = panel.querySelector(".trail");
    if (seat) {
      showCards(cards, seat.cards, { small: true });
      const { total } = game.handValue(seat.cards);
      trail.innerHTML = seat.actions.map((a) => `<span class="chip">${label(a)}</span>`).join("")
        + `<span class="chip">${total > 21 ? `${total}, dépassé` : `total ${total}`}</span>`;
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
  const rows = [{ id: "you", name: "Vous", exact: null }, ...models.map((m) => ({ id: m.id, name: m.name, exact: m.exactGain }))];
  const played = scores.you.n > 0;
  const best = Math.max(...rows.map((r) => scores[r.id].sum));
  $("score-rows").replaceChildren(...rows.map((r) => {
    const { sum, n } = scores[r.id];
    const tr = document.createElement("tr");
    if (played && sum === best && rows.some((o) => scores[o.id].sum !== best)) tr.className = "leader";
    const cls = sum > 0 ? "positive" : sum < 0 ? "negative" : "";
    tr.innerHTML = `<td>${r.name}</td><td class="${cls}">${fmt(sum, 0)}</td><td>${n ? fmt(sum / n) : "—"}</td>
      <td>${r.exact === null ? "—" : fmt(r.exact, 4)}</td>`;
    return tr;
  }));
  $("score-note").textContent = played
    ? `${scores.you.n} partie${scores.you.n > 1 ? "s" : ""}. Sur si peu de parties, la chance domine : une partie a un écart-type de 0,95, `
      + "et il faut des millions de parties pour départager les modèles. Le gain exact, lui, est calculé sans hasard."
    : "Le gain exact attendu est calculé sans simulation, par l'oracle du projet.";
}

function renderHint() {
  const panel = $("hint-panel");
  if (panel.hidden || phase !== "playing") return;
  const obs = game.observation(you, round);
  panel.replaceChildren(...models.map((model) => {
    const a = game.advice(model, obs);
    const row = document.createElement("div");
    row.className = "hint-row";
    row.innerHTML = `<span><strong>${model.name}</strong> · ${label(a.action)}</span>
      <span class="values">STAND ${fmt(a.stand)} · HIT ${fmt(a.hit)}</span>`;
    return row;
  }));
}

function setStatus(html) { $("status").innerHTML = html; }

function renderCount() {
  const n = scores.you.n;
  $("round-count").textContent = phase === "playing" ? `Donne ${n + 1}` : n ? `${n} donne${n > 1 ? "s" : ""} jouée${n > 1 ? "s" : ""}` : "Aucune donne";
}

function render() { renderTable(); renderSeats(); renderScores(); renderHint(); renderCount(); }

function startRound() {
  round = game.newRound();
  you = game.newSeat();
  game.deal(round, you);
  phase = "playing"; conform = []; results = {};
  $("player-cards").replaceChildren();
  $("dealer-cards").replaceChildren(cardElement(round.dealer[0]), cardElement(null, { hidden: true }));
  $("dealer-total").classList.remove("bust");
  if (game.handValue(you.cards).total === 21) { game.stand(you); return finish(); }
  render();
  const obs = game.observation(you, round);
  setStatus(`Vous avez <strong>${obs.total}</strong> face à ${round.dealer[0].rank === "A" ? "un <strong>as</strong>" : `un <strong>${round.dealer[0].value}</strong>`}. Tirer ou rester ?`);
}

function act(action) {
  if (phase !== "playing") return;
  conform.push(game.advice(optimal, game.observation(you, round)).action === action);
  if (action === "HIT") game.hit(round, you); else game.stand(you);
  if (!you.done && game.handValue(you.cards).total === 21) game.stand(you);  // à 21, tirer ne peut que perdre
  if (you.done) return finish();
  render();
  setStatus(`Vous avez <strong>${game.handValue(you.cards).total}</strong>. Tirer ou rester ?`);
}

function finish() {
  phase = "done";
  for (const model of models) {
    const seat = game.newSeat();
    game.playModel(round, seat, model);
    results[model.id] = seat;
  }
  game.playDealer(round);
  const gain = game.outcome(you, round);
  scores.you.sum += gain; scores.you.n += 1;
  for (const model of models) {
    results[model.id].gain = game.outcome(results[model.id], round);
    scores[model.id].sum += results[model.id].gain; scores[model.id].n += 1;
  }
  $("hint-panel").hidden = true; $("hint").setAttribute("aria-expanded", "false");
  render();
  const verdict = gain > 0 ? "victoire (+1)" : gain < 0 ? "défaite (−1)" : "égalité (0)";
  const dealer = game.handValue(round.dealer).total;
  const good = conform.filter(Boolean).length;
  setStatus(`<strong>${verdict}</strong> · croupier ${dealer > 21 ? `${dealer}, dépassé` : dealer}`
    + (conform.length ? ` · ${good} décision${good > 1 ? "s" : ""} sur ${conform.length} conforme${conform.length > 1 ? "s" : ""} à la stratégie optimale.` : " · 21 servi."));
}

function toggleHint() {
  if (phase !== "playing") return;
  const panel = $("hint-panel");
  panel.hidden = !panel.hidden;
  $("hint").setAttribute("aria-expanded", String(!panel.hidden));
  renderHint();
}

$("deal").addEventListener("click", startRound);
$("hit").addEventListener("click", () => act("HIT"));
$("stand").addEventListener("click", () => act("STAND"));
$("hint").addEventListener("click", toggleHint);
$("open-rules").addEventListener("click", () => $("rules").showModal());
$("reset").addEventListener("click", () => { Object.values(scores).forEach((s) => { s.sum = 0; s.n = 0; }); renderScores(); renderCount(); });
document.addEventListener("keydown", (event) => {
  if ($("rules").open || event.metaKey || event.ctrlKey || event.altKey) return;
  const key = event.key.toLowerCase();
  if (key === "h") act("HIT");
  else if (key === "s") act("STAND");
  else if (key === "c") toggleHint();
  else if ((key === " " || key === "n") && phase !== "playing") { event.preventDefault(); startRound(); }
});

render();
