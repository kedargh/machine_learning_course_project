const names = {logistic_regression: 'Logistic Regression', decision_tree: 'Decision Tree', random_forest: 'Random Forest', xgboost: 'XGBoost'};
const $ = id => document.getElementById(id);
const percent = value => `${(value * 100).toFixed(1)}%`;
const fields = [
  ['age', 'Age', 42, 18, 120], ['job', 'Occupation', ['management','technician','entrepreneur','blue-collar','retired','admin.','services','self-employed','unemployed','housemaid','student','unknown']],
  ['marital','Marital status',['married','single','divorced']], ['education','Education',['secondary','primary','tertiary','unknown']],
  ['balance','Account balance',1500], ['default','Credit in default',['no','yes','unknown']], ['housing','Housing loan',['no','yes','unknown']], ['loan','Personal loan',['no','yes','unknown']],
  ['contact','Contact channel',['cellular','telephone','unknown']], ['day','Contact day',15,1,31], ['month','Contact month',['may','jan','feb','mar','apr','jun','jul','aug','sep','oct','nov','dec']],
  ['campaign','Contacts this campaign',1,1], ['pdays','Days since prior contact',-1,-1], ['previous','Previous contacts',0,0], ['poutcome','Prior campaign outcome',['unknown','failure','other','success']]
];
fields.forEach(([key, title, value, min, max], index) => {
  const label = document.createElement('label');
  label.textContent = title;
  const input = document.createElement(Array.isArray(value) ? 'select' : 'input');
  input.name = key;
  if (Array.isArray(value)) value.forEach(item => input.add(new Option(item.replaceAll('-', ' '), item)));
  else { input.type = 'number'; input.value = value; input.step = key === 'balance' ? 'any' : '1'; if (min !== undefined) input.min = min; if (max !== undefined) input.max = max; }
  input.required = true;
  label.append(input);
  $(index < 8 ? 'profile-fields' : 'campaign-fields').append(label);
});
async function request(url, options) {
  const response = await fetch(url, options);
  let body;
  try { body = await response.json(); } catch { throw new Error('The server returned an unexpected response.'); }
  if (!response.ok) throw new Error(body.error || 'Request failed.');
  return body;
}
function showError(id, message) { $(id).textContent = message; $(id).hidden = !message; }
async function loadDashboard() {
  $('refresh').disabled = true;
  showError('dashboard-error', '');
  try {
    const {scores, metadata} = await request('/api/dashboard');
    $('connection').textContent = 'Backend connected';
    const sorted = [...scores].sort((a,b) => b.pr_auc - a.pr_auc);
    $('scores').replaceChildren();
    sorted.forEach((score, index) => {
      const row = document.createElement('tr');
      if (!index) row.className = 'winner';
      const label = document.createElement('td'); label.textContent = names[score.model] || score.model;
      if (!index) { const badge = document.createElement('span'); badge.className = 'best'; badge.textContent = 'BEST PR-AUC'; label.append(badge); }
      row.append(label);
      ['accuracy','precision','recall','f1_score','roc_auc','pr_auc'].forEach(key => {const cell = document.createElement('td'); cell.textContent = percent(score[key]); row.append(cell);});
      $('scores').append(row);
    });
    const stats = [
      ['BEST MODEL', names[sorted[0]?.model] || '—', 'Highest test PR-AUC'],
      ['CUSTOMER RECORDS', metadata.ingestion.rows_after_deduplication.toLocaleString(), 'After duplicate removal'],
      ['TEST SET', metadata.test_rows.toLocaleString(), `${Math.round(metadata.test_size * 100)}% held out for evaluation`],
      ['SUBSCRIPTION RATE', percent(metadata.training_positive_rate), 'Positive class in training data']
    ];
    $('summary').replaceChildren();
    stats.forEach(([title,value,note]) => { const card = document.createElement('article'); card.className = 'stat'; [['div',title,'stat-label'],['strong',value,''],['small',note,'']].forEach(([tag,text,cls]) => {const el = document.createElement(tag); el.textContent = text; el.className = cls; card.append(el);}); $('summary').append(card); });
    $('threshold').textContent = `Threshold ${percent(metadata.decision_threshold)}`;
  } catch (error) {
    $('connection').textContent = 'Results unavailable';
    showError('dashboard-error', error.message);
    $('scores').replaceChildren(); $('summary').replaceChildren();
  } finally { $('refresh').disabled = false; }
}
function clearPrediction() {
  $('result-title').textContent = 'Ready to explore.';
  $('result-description').textContent = 'Run a prediction for the current customer and classifier.';
  $('probability').replaceChildren(document.createTextNode('—'), Object.assign(document.createElement('span'), {textContent:'%'}));
  $('meter-fill').style.width = '0%'; $('decision').textContent = 'Awaiting prediction';
}
$('prediction-form').addEventListener('input', clearPrediction);
$('example').addEventListener('click', () => { $('prediction-form').reset(); clearPrediction(); showError('prediction-error',''); });
$('refresh').addEventListener('click', loadDashboard);
$('prediction-form').addEventListener('submit', async event => {
  event.preventDefault(); showError('prediction-error',''); clearPrediction();
  const features = Object.fromEntries(new FormData(event.target)); const model = features.model; delete features.model;
  $('predict').disabled = true; $('decision').textContent = 'Evaluating customer…';
  // Freeze fields so the displayed result always matches the submitted profile.
  const controls = [...event.target.elements, $('example')]; controls.forEach(el => el.disabled = true);
  try {
    const result = await request('/api/predict', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({model, features})});
    $('result-title').textContent = result.prediction === 'yes' ? 'Subscription predicted.' : 'No subscription predicted.';
    $('result-description').textContent = `${names[model]} · Current customer profile`;
    $('probability').replaceChildren(document.createTextNode((result.probability * 100).toFixed(1)), Object.assign(document.createElement('span'), {textContent:'%'}));
    $('meter-fill').style.width = percent(result.probability);
    $('threshold').textContent = `Threshold ${percent(result.threshold)}`;
    $('decision').textContent = result.prediction === 'yes' ? 'Above the subscription decision threshold' : 'Below the subscription decision threshold';
  } catch (error) { showError('prediction-error',error.message); $('decision').textContent = 'Prediction unavailable'; }
  finally { controls.forEach(el => el.disabled = false); }
});
loadDashboard();
