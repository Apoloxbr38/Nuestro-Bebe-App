// ===============================
//  APP.JS A PRUEBA DE BALAS 😈
// ===============================

// Usa el mismo origen del backend que sirve /app/
const API = window.location.origin;

// ---------- Utilidades ----------
const log = (...a) => console.log("[APP]", ...a);
const warn = (...a) => console.warn("[APP]", ...a);
const err  = (...a) => console.error("[APP]", ...a);

function $(id){ return document.getElementById(id); }
function opt(v){ const o=document.createElement("option"); o.value=v; o.textContent=v; return o; }
function asPct(x){ return Math.max(0, Math.min(100, Math.round(Number(x)*100))); }

// Fetch helpers SIN AbortController (evita “signal is aborted”)
async function getJSON(url){
  log("GET", url);
  try{
    const r = await fetch(url, { cache: "no-store" });
    if(!r.ok){
      const body = await r.text().catch(()=> "");
      throw new Error(`HTTP ${r.status} @ ${url}\n${body}`);
    }
    return await r.json();
  }catch(e){
    err("getJSON ERROR:", e);
    return Promise.reject(e);
  }
}

async function postJSON(url, body){
  log("POST", url, body);
  try{
    const r = await fetch(url, {
      method: "POST",
      headers: {"Content-Type":"application/json"},
      body: JSON.stringify(body || {})
    });
    if(!r.ok){
      const bodyTxt = await r.text().catch(()=> "");
      throw new Error(`HTTP ${r.status} @ ${url}\n${bodyTxt}`);
    }
    return await r.json();
  }catch(e){
    err("postJSON ERROR:", e);
    return Promise.reject(e);
  }
}

// ---------- Donuts (Chart.js opcional) ----------
const Charts = { x12: null, ou: null, btts: null };

function makeOrUpdateDonut(canvasId, values, labels){
  const canvas = $(canvasId);
  if(!canvas){ warn("No canvas:", canvasId); return; }
  if(typeof Chart === "undefined"){ warn("Chart.js no está cargado. Saltando donuts."); return; }

  const data = { labels, datasets: [{ data: values, backgroundColor: ['#60a5fa','#a78bfa','#f472b6'], hoverOffset: 6, borderWidth: 0 }] };
  const options = {
    responsive: false,
    plugins: {
      legend: { position: 'bottom', labels: { color: '#e9e9f1' } },
      tooltip: { callbacks: { label: (c)=> `${c.label}: ${c.raw}%` } }
    },
    animation: { duration: 400 },
    cutout: '60%'
  };

  const key = (canvasId==='chart-1x2'?'x12':canvasId==='chart-ou'?'ou':'btts');
  if(Charts[key]){
    Charts[key].data.labels = labels;
    Charts[key].data.datasets[0].data = values;
    Charts[key].update();
  }else{
    Charts[key] = new Chart(canvas.getContext("2d"), { type: 'doughnut', data, options });
  }
}

// ---------- Estado ----------
let lastPayload = null;

// ---------- Ligas & Equipos ----------
async function loadLeagues(){
  log("Cargando ligas…");
  const sel = $("league");
  if(!sel){ warn("No hay #league en el DOM, saltando."); return []; }

  let leagues = [];
  try{
    const data = await getJSON(`${API}/leagues`);
    leagues = data.leagues || [];
  }catch(e){
    err("Fallo /leagues:", e);
  }

  sel.innerHTML = "";
  leagues.forEach(lg => sel.appendChild(opt(lg)));
  log("Ligas:", leagues);
  return leagues;
}

async function loadTeamsForLeague(league){
  const sH = $("home"), sA = $("away");
  if(!sH || !sA){ warn("Faltan selects #home o #away."); return; }
  if(!league){ warn("Liga vacía."); return; }

  log("Cargando equipos de", league);
  try{
    const data = await getJSON(`${API}/teams?league=${encodeURIComponent(league)}`);
    const teams = data.teams || [];
    sH.innerHTML = ""; sA.innerHTML = "";
    teams.forEach(t => { sH.appendChild(opt(t)); sA.appendChild(opt(t)); });
    if(teams.length>1){ sA.value = teams[1]; }
    log(`Equipos (${league}):`, teams.length);
  }catch(e){
    err("Fallo /teams:", e);
    alert("No pude cargar equipos. ¿Backend ok?");
  }
}

async function init(){
  log("Init UI…");

  // Test rápido de conectividad
  try{
    const ping = await getJSON(`${API}/health`);
    log("/health OK:", ping);
  }catch(e){
    err("/health FALLÓ:", e);
    alert("No pude hablar con el backend (/health). Abre http://127.0.0.1:8000/health para verificar.");
  }

  // Intenta cargar ligas; si no hay, cae a teams globales
  const leagues = await loadLeagues();
  const sel = $("league");
  if(sel && leagues.length){
    await loadTeamsForLeague(sel.value || leagues[0]);
    sel.onchange = ()=> loadTeamsForLeague(sel.value);
  } else {
    // fallback dataset antiguo
    try{
      const data = await getJSON(`${API}/teams`);
      const teams = data.teams || [];
      const sH = $("home"), sA = $("away");
      if(sH && sA){
        sH.innerHTML = ""; sA.innerHTML = "";
        teams.forEach(t => { sH.appendChild(opt(t)); sA.appendChild(opt(t)); });
        if(teams.length>1){ sA.value = teams[1]; }
      }
      log("Fallback equipos globales:", teams.length);
    }catch(e){
      err("Fallo /teams (fallback):", e);
      alert("No pude cargar ligas/equipos. ¿Backend ok?");
    }
  }
}

// ---------- Explicaciones ----------
async function openExplain(kind){
  const home = $("home")?.value;
  const away = $("away")?.value;
  if(!home || !away){ alert("Elige dos equipos, bombón 😘"); return; }

  try{
    const data = await getJSON(`${API}/explain?home=${encodeURIComponent(home)}&away=${encodeURIComponent(away)}`);
    const reasons = (data.reasons||{})[kind] || [];
    alert(`${kind.toUpperCase()} — razones:\n- ${reasons.join("\n- ")}`);
  }catch(e){
    err("Fallo /explain:", e);
    alert("No pude obtener explicación.");
  }
}

// ---------- Render ----------
function paintChip(id, src){
  const el = $(id);
  if(!el) return;
  const map = { ml:"ML", xg:"xG", poisson:"Poisson" };
  el.textContent = map[src] ?? src ?? "–";
  el.classList.remove("ml","xg","poisson");
  if(src) el.classList.add(src);
}

function renderBars1x2(ph, pd, pa){
  if($("bar-home")) $("bar-home").style.width = ph+'%';
  if($("bar-draw")) $("bar-draw").style.width = pd+'%';
  if($("bar-away")) $("bar-away").style.width = pa+'%';
  if($("lab-home")) $("lab-home").textContent = ph+'%';
  if($("lab-draw")) $("lab-draw").textContent = pd+'%';
  if($("lab-away")) $("lab-away").textContent = pa+'%';
  makeOrUpdateDonut('chart-1x2', [ph, pd, pa], ['Local', 'Empate', 'Visitante']);
}

function renderBarsOU(pOver, pUnder){
  if($("bar-ou-over")) $("bar-ou-over").style.width = pOver+'%';
  if($("bar-ou-under")) $("bar-ou-under").style.width = pUnder+'%';
  if($("lab-ou-over")) $("lab-ou-over").textContent = pOver+'%';
  if($("lab-ou-under")) $("lab-ou-under").textContent = pUnder+'%';
  makeOrUpdateDonut('chart-ou', [pOver, pUnder], ['Over 2.5', 'Under 2.5']);
}

function renderBarsBTTS(pYes, pNo){
  if($("bar-btts-yes")) $("bar-btts-yes").style.width = pYes+'%';
  if($("bar-btts-no")) $("bar-btts-no").style.width  = pNo+'%';
  if($("lab-btts-yes")) $("lab-btts-yes").textContent = pYes+'%';
  if($("lab-btts-no")) $("lab-btts-no").textContent  = pNo+'%';
  makeOrUpdateDonut('chart-btts', [pYes, pNo], ['Sí', 'No']);
}

function renderScorelines(listId, arr){
  const list = $(listId);
  if(!list) return;
  list.innerHTML = "";
  (arr||[]).forEach(s => {
    const li = document.createElement("li");
    li.textContent = `${s.score} — ${(s.prob*100).toFixed(2)}%`;
    list.appendChild(li);
  });
}

// ---------- Predicción ----------
async function predict(){
  log("predict() llamado");
  const home = $("home")?.value;
  const away = $("away")?.value;
  log("selección:", {home, away});

  if(!home || !away){ alert("Elige dos equipos, bombón 😘"); return; }
  if(home === away){ alert("Elige equipos distintos, bombón 😘"); return; }

  const url = `${API}/predict?home=${encodeURIComponent(home)}&away=${encodeURIComponent(away)}`;
  log("GET", url);

  try{
    const data = await getJSON(url);
    log("/predict OK:", data);
    lastPayload = data;

    // 1X2
    const ph = asPct(data.p_home), pd = asPct(data.p_draw), pa = asPct(data.p_away);
    renderBars1x2(ph, pd, pa);

    // xG
    if($("xg-home")) $("xg-home").textContent = data.exp_goals_home;
    if($("xg-away")) $("xg-away").textContent = data.exp_goals_away;
    const xgt = (typeof data.exp_goals_total === 'number')
      ? data.exp_goals_total
      : (Number(data.exp_goals_home||0) + Number(data.exp_goals_away||0));
    if($("xg-total")) $("xg-total").textContent = Number(xgt).toFixed(3);

    // OU
    const pOver = asPct(data.ou_over25 ?? 0);
    const pUnder = asPct(data.ou_under25 ?? (1-(data.ou_over25??0)));
    renderBarsOU(pOver, pUnder);

    // BTTS
    const pYes = asPct(data.btts_yes ?? 0);
    const pNo  = asPct(data.btts_no  ?? (1-(data.btts_yes??0)));
    renderBarsBTTS(pYes, pNo);

    // Marcadores
    renderScorelines("scorelines", data.top_scorelines || []);

    // Badges de fuente
    paintChip("chip-1x2", data.src_1x2);
    paintChip("chip-ou",  data.src_ou25);
    paintChip("chip-btts",data.src_btts);

    if($("results")) $("results").classList.remove("hidden");
  }catch(e){
    err("predict() fallo:", e);
    alert("Fallo /predict:\n" + (e?.message || e));
  }
}

// ---------- Value Finder ----------
function impliedProb(odds){ return odds>0 ? 1/odds : NaN; }
function edge(ourP, odds){ const imp = impliedProb(odds); return isNaN(imp) ? NaN : (ourP - imp); }

function showValue(listId, items){
  const ul = $(listId);
  if(!ul) return;
  ul.innerHTML = "";
  items.forEach(x=>{
    const li = document.createElement("li");
    const pct = (x.edge*100).toFixed(2);
    li.textContent = `${x.label}: edge ${pct}% (nuestra p=${(x.our*100).toFixed(1)}%, implícita=${(x.imp*100).toFixed(1)}%)`;
    li.className = (x.edge>0 ? 'good' : 'bad');
    ul.appendChild(li);
  });
}

function calcValue1x2(){
  if(!lastPayload){ alert("Primero haz una predicción."); return; }
  const {p_home,p_draw,p_away} = lastPayload;
  const oh = parseFloat($("odds-h")?.value || "");
  const od = parseFloat($("odds-d")?.value || "");
  const oa = parseFloat($("odds-a")?.value || "");
  const items = [];
  if(oh) items.push({label:'Local',    our:p_home, imp:impliedProb(oh), edge:edge(p_home,oh)});
  if(od) items.push({label:'Empate',   our:p_draw, imp:impliedProb(od), edge:edge(p_draw,od)});
  if(oa) items.push({label:'Visitante',our:p_away, imp:impliedProb(oa), edge:edge(p_away,oa)});
  showValue("value-1x2", items);
}

function calcValueOU(){
  if(!lastPayload){ alert("Primero haz una predicción."); return; }
  const {ou_over25,ou_under25} = lastPayload;
  const oov = parseFloat($("odds-over")?.value || "");
  const oud = parseFloat($("odds-under")?.value || "");
  const items = [];
  if(oov) items.push({label:'Over 2.5',  our:ou_over25, imp:impliedProb(oov), edge:edge(ou_over25,oov)});
  if(oud) items.push({label:'Under 2.5', our:ou_under25, imp:impliedProb(oud), edge:edge(ou_under25,oud)});
  showValue("value-ou", items);
}

function calcValueBTTS(){
  if(!lastPayload){ alert("Primero haz una predicción."); return; }
  const {btts_yes,btts_no} = lastPayload;
  const oy = parseFloat($("odds-btts-yes")?.value || "");
  const on = parseFloat($("odds-btts-no")?.value  || "");
  const items = [];
  if(oy) items.push({label:'BTTS Sí', our:btts_yes, imp:impliedProb(oy), edge:edge(btts_yes,oy)});
  if(on) items.push({label:'BTTS No', our:btts_no,  imp:impliedProb(on), edge:edge(btts_no,on)});
  showValue("value-btts", items);
}

// ---------- Recarga multi-liga ----------
async function reloadMulti(){
  try{
    // Ajusta a tu gusto:
    const leagues = ['SP1','E0','I1'];            // LaLiga, Premier, Serie A
    const start_years = [2024, 2023, 2022, 2021]; // 4 temporadas atrás
    const j = await postJSON(`${API}/reload_multi`, { leagues, start_years });
    alert(`Datos recargados: ${j.rows} filas\nLigas: ${j.leagues.join(', ')}\nAños: ${j.years.join(', ')}`);
    await init();
  }catch(e){
    err("Fallo reload_multi:", e);
    alert("No pude recargar datos multi-liga.");
  }
}

// ---------- Enganches robustos ----------
function bindPredictButton(){
  const b = $("btn") || $("btn-predict");
  if(!b){ warn("No encontré botón #btn. Reintentando…"); setTimeout(bindPredictButton, 300); return; }
  log("Botón predecir enganchado ✅");
  b.addEventListener("click", (e)=>{ e.preventDefault(); predict(); });
}

function bindOthers(){
  $("reload")?.addEventListener("click", reloadMulti);
  $("btn-value-1x2")?.addEventListener("click", calcValue1x2);
  $("btn-value-ou")?.addEventListener("click",  calcValueOU);
  $("btn-value-btts")?.addEventListener("click", calcValueBTTS);

  $("exp-1x2")?.addEventListener("click", ()=>openExplain("1x2"));
  $("exp-ou")?.addEventListener("click",  ()=>openExplain("ou25"));
  $("exp-btts")?.addEventListener("click",()=>openExplain("btts"));
}

// ---------- Bootstrap ----------
document.addEventListener("DOMContentLoaded", async () => {
  try{
    bindPredictButton();
    bindOthers();
    await init();
    log("App lista 💋");
  }catch(e){
    err("Error inicializando app:", e);
    alert("Se rompió el JS al iniciar. Revisa la consola (F12 > Console).");
  }
});
