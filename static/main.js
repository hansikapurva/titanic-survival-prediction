// ── Utility ──────────────────────────────────────────────────────────────
const API = '';  // empty = same origin (Flask serves both)

function scrollTo(id) {
  document.getElementById(id).scrollIntoView({ behavior: 'smooth' });
}

// Active nav link on scroll
const sections = ['home', 'predict', 'eda', 'model', 'batch'];
window.addEventListener('scroll', () => {
  let cur = 'home';
  for (const id of sections) {
    const el = document.getElementById(id);
    if (el && el.getBoundingClientRect().top <= 80) cur = id;
  }
  document.querySelectorAll('.nav-link').forEach(l => {
    l.classList.toggle('active', l.dataset.section === cur);
  });
});

// ── Fetch stats & init charts ────────────────────────────────────────────
let statsData = null;

async function loadStats() {
  try {
    const res = await fetch(`${API}/api/stats`);
    statsData = await res.json();
    populateEDA(statsData);
    populateModel(statsData);
    document.getElementById('hero-acc').textContent = statsData.accuracy + '%';
  } catch (e) {
    console.error('Failed to load stats', e);
  }
}

function populateEDA(data) {
  const eda = data.eda;

  // Strip
  document.getElementById('strip-total').textContent = eda.total;
  document.getElementById('strip-surv').textContent = eda.survived;
  document.getElementById('strip-rate').textContent = eda.survival_rate + '%';

  const PALETTE = {
    navy: '#0a1628', ocean: '#1565c0', gold: '#f0a500',
    teal: '#00897b', coral: '#e64a19', green: '#388e3c', amber: '#f57c00',
    border: '#e2e8f0', text2: '#4a5568'
  };

  const chartDefaults = {
    plugins: { legend: { display: false } },
    scales: {
      x: { grid: { color: PALETTE.border }, ticks: { color: PALETTE.text2, font: { size: 12 } } },
      y: { grid: { color: PALETTE.border }, ticks: { color: PALETTE.text2, font: { size: 12 } }, beginAtZero: true }
    }
  };

  // Survival distribution (bar)
  new Chart(document.getElementById('chart-survival'), {
    type: 'bar',
    data: {
      labels: ['Did not survive', 'Survived'],
      datasets: [{
        data: [eda.not_survived, eda.survived],
        backgroundColor: [PALETTE.coral, PALETTE.teal],
        borderRadius: 8, borderSkipped: false
      }]
    },
    options: { ...chartDefaults, plugins: { legend: { display: false } } }
  });

  // Gender survival (grouped bar)
  const gd = eda.gender_survival;
  new Chart(document.getElementById('chart-gender'), {
    type: 'bar',
    data: {
      labels: ['Female', 'Male'],
      datasets: [
        { label: 'Survival rate (%)', data: [gd.female || 0, gd.male || 0], backgroundColor: [PALETTE.ocean, PALETTE.gold], borderRadius: 8, borderSkipped: false }
      ]
    },
    options: { ...chartDefaults, plugins: { legend: { display: false } } }
  });

  // Class survival
  const cd = eda.class_survival;
  new Chart(document.getElementById('chart-class'), {
    type: 'bar',
    data: {
      labels: ['1st Class', '2nd Class', '3rd Class'],
      datasets: [{
        data: [cd['1'] || 0, cd['2'] || 0, cd['3'] || 0],
        backgroundColor: [PALETTE.teal, PALETTE.ocean, PALETTE.coral],
        borderRadius: 8, borderSkipped: false
      }]
    },
    options: { ...chartDefaults }
  });

  // Age bins (bar)
  const ab = eda.age_bins;
  new Chart(document.getElementById('chart-age'), {
    type: 'bar',
    data: {
      labels: ['0–16', '16–32', '32–48', '48–64', '64+'],
      datasets: [{
        data: [ab.children, ab.young, ab.adult, ab.middle, ab.senior],
        backgroundColor: PALETTE.amber,
        borderRadius: 8, borderSkipped: false
      }]
    },
    options: { ...chartDefaults }
  });
}

function populateModel(data) {
  document.getElementById('m-accuracy').textContent = data.accuracy + '%';

  const r = data.report;
  if (r && r['1']) {
    document.getElementById('m-precision').textContent = Math.round(r['1'].precision * 100) + '%';
    document.getElementById('m-recall').textContent = Math.round(r['1'].recall * 100) + '%';
  }

  // Confusion matrix
  const cm = data.confusion_matrix;
  const tn = cm[0][0], fp = cm[0][1], fn = cm[1][0], tp = cm[1][1];
  document.getElementById('confusion-matrix').innerHTML = `
    <div class="cm-cell tp"><span class="cm-num">${tp}</span><span class="cm-label">True Positive</span></div>
    <div class="cm-cell fp"><span class="cm-num">${fp}</span><span class="cm-label">False Positive</span></div>
    <div class="cm-cell fn"><span class="cm-num">${fn}</span><span class="cm-label">False Negative</span></div>
    <div class="cm-cell tn"><span class="cm-num">${tn}</span><span class="cm-label">True Negative</span></div>
  `;

  // Feature importance chart
  const fi = data.feature_importances.slice(0, 8);
  new Chart(document.getElementById('chart-importance'), {
    type: 'bar',
    data: {
      labels: fi.map(f => f.feature),
      datasets: [{
        data: fi.map(f => f.importance),
        backgroundColor: '#1565c0',
        borderRadius: 6, borderSkipped: false
      }]
    },
    options: {
      indexAxis: 'y',
      plugins: { legend: { display: false } },
      scales: {
        x: { grid: { color: '#e2e8f0' }, ticks: { color: '#4a5568', font: { size: 11 } }, beginAtZero: true },
        y: { grid: { display: false }, ticks: { color: '#1a2332', font: { size: 12 } } }
      }
    }
  });
}

// ── Prediction ────────────────────────────────────────────────────────────
async function runPredict() {
  const btn = document.getElementById('predict-btn');
  btn.classList.add('loading');
  btn.querySelector('.btn-text').textContent = 'Predicting...';

  const payload = {
    pclass: document.getElementById('p-pclass').value,
    sex: document.getElementById('p-sex').value,
    age: document.getElementById('p-age').value,
    fare: document.getElementById('p-fare').value,
    sibsp: document.getElementById('p-sibsp').value,
    parch: document.getElementById('p-parch').value,
    embarked: document.getElementById('p-embarked').value
  };

  try {
    const res = await fetch(`${API}/api/predict`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    const data = await res.json();
    showResult(data, payload);
  } catch (e) {
    alert('Error connecting to backend. Make sure Flask is running!');
  } finally {
    btn.classList.remove('loading');
    btn.querySelector('.btn-text').textContent = 'Predict my survival';
  }
}

function showResult(data, inputs) {
  document.getElementById('result-placeholder').style.display = 'none';
  const rc = document.getElementById('result-content');
  rc.style.display = 'flex';

  const survived = data.survived;
  const prob = data.probability;

  document.getElementById('result-emoji').textContent = survived ? '🛟' : '💀';
  const verdict = document.getElementById('result-verdict');
  verdict.textContent = survived ? 'Survived!' : 'Did not survive';
  verdict.className = 'result-verdict ' + (survived ? 'survived' : 'died');
  document.getElementById('result-prob').textContent = prob.toFixed(1) + '%';

  const fill = document.getElementById('prob-fill');
  fill.className = 'prob-fill ' + (survived ? 'survived' : 'died');
  setTimeout(() => { fill.style.width = prob + '%'; }, 100);

  // Factors
  const factors = [];
  if (inputs.sex === 'female') factors.push('Female passengers had priority in lifeboats');
  if (inputs.sex === 'male') factors.push('Male passengers had lower survival rates (~19%)');
  if (parseInt(inputs.pclass) === 1) factors.push('1st class passengers had better access to lifeboats');
  if (parseInt(inputs.pclass) === 3) factors.push('3rd class passengers were located furthest from lifeboats');
  if (parseFloat(inputs.age) < 16) factors.push('Children were given evacuation priority');
  const fs = parseInt(inputs.sibsp) + parseInt(inputs.parch) + 1;
  if (fs >= 2 && fs <= 4) factors.push('Traveling with a small family improved survival odds');
  if (fs > 5) factors.push('Large family groups had difficulty evacuating together');

  document.getElementById('factor-list').innerHTML =
    factors.slice(0, 3).map(f => `<div class="factor-item">${f}</div>`).join('');
}

function resetPredict() {
  document.getElementById('result-placeholder').style.display = 'flex';
  document.getElementById('result-content').style.display = 'none';
}

// ── Batch predict ─────────────────────────────────────────────────────────
async function runBatchPredict() {
  let passengers;
  try {
    passengers = JSON.parse(document.getElementById('batch-json').value);
  } catch (e) {
    alert('Invalid JSON! Please check your input format.');
    return;
  }

  const resultsDiv = document.getElementById('batch-results');
  resultsDiv.innerHTML = '<div class="batch-empty">Running predictions...</div>';

  try {
    const res = await fetch(`${API}/api/batch_predict`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ passengers })
    });
    const results = await res.json();

    if (!results.length) {
      resultsDiv.innerHTML = '<div class="batch-empty">No results returned.</div>';
      return;
    }

    resultsDiv.innerHTML = results.map(r => `
      <div class="batch-result-item">
        <span class="batch-pnum">Passenger ${r.passenger}</span>
        ${r.error
          ? `<span style="color:#e64a19;font-size:13px">Error: ${r.error}</span>`
          : `<span class="batch-verdict ${r.survived ? 'survived' : 'died'}">${r.survived ? '✓ Survived' : '✗ Did not survive'}</span>
             <span class="batch-prob">${r.probability.toFixed(1)}% survival probability</span>`
        }
      </div>`).join('');
  } catch (e) {
    resultsDiv.innerHTML = '<div class="batch-empty">Error connecting to backend.</div>';
  }
}

function clearBatch() {
  document.getElementById('batch-results').innerHTML = '<div class="batch-empty">Results will appear here after running a prediction.</div>';
}

// ── Sample passengers table ───────────────────────────────────────────────
async function loadSamplePassengers() {
  try {
    const res = await fetch(`${API}/api/sample_passengers`);
    const passengers = await res.json();
    const tbody = document.getElementById('sample-tbody');
    tbody.innerHTML = passengers.map((p, i) => `
      <tr>
        <td>${p.Pclass}</td>
        <td style="text-transform:capitalize">${p.Sex}</td>
        <td>${p.Age}</td>
        <td>${p.SibSp}</td>
        <td>${p.Parch}</td>
        <td>£${p.Fare.toFixed(2)}</td>
        <td>${p.Embarked === 'S' ? 'Southampton' : p.Embarked === 'C' ? 'Cherbourg' : 'Queenstown'}</td>
        <td><span class="badge ${p.ActualSurvived ? 'badge-survived' : 'badge-died'}">${p.ActualSurvived ? 'Yes' : 'No'}</span></td>
        <td>
          <button class="btn-table btn-predict-row" onclick="loadIntoPredict(${i})">Predict</button>
          <button class="btn-table btn-load-row" onclick="loadIntoBatch(${i})">Batch load</button>
        </td>
      </tr>
    `).join('');
    window._samplePassengers = passengers;
  } catch (e) {
    console.error('Failed to load sample passengers');
  }
}

function loadIntoPredict(i) {
  const p = window._samplePassengers[i];
  document.getElementById('p-pclass').value = p.Pclass;
  document.getElementById('p-sex').value = p.Sex;
  document.getElementById('p-age').value = p.Age;
  document.getElementById('p-fare').value = p.Fare.toFixed(2);
  document.getElementById('p-sibsp').value = p.SibSp;
  document.getElementById('p-parch').value = p.Parch;
  document.getElementById('p-embarked').value = p.Embarked;
  document.getElementById('predict').scrollIntoView({ behavior: 'smooth' });
  resetPredict();
}

function loadIntoBatch(i) {
  const p = { ...window._samplePassengers[i] };
  delete p.ActualSurvived;
  document.getElementById('batch-json').value = JSON.stringify([p], null, 2);
  document.getElementById('batch').scrollIntoView({ behavior: 'smooth' });
}

// ── Init ─────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  loadStats();
  loadSamplePassengers();
});
