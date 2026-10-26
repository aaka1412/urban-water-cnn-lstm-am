import { Component, OnInit, computed, inject, signal } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { RouterLink } from '@angular/router';
import { ChartConfiguration } from 'chart.js';
import { COLORS, DChart, Dash, baseOptions, errText, fmt, when } from './core';
import { StatusChip } from './pages-user';

const PALETTE = [COLORS.navy, COLORS.blue, COLORS.sky, COLORS.teal, COLORS.amber, COLORS.grey];

// ---------------------------------------------------------------- Model comparison
@Component({
  selector: 'd-compare',
  standalone: true,
  imports: [FormsModule, DChart, RouterLink],
  template: `
  <div class="d-pagehead"><h2>Model Comparison</h2>
    <label class="d-field">Best model by<select [ngModel]="metric()" (ngModelChange)="metric.set($event); load()"><option value="rmse">RMSE (lower is better)</option><option value="mae">MAE (lower is better)</option><option value="mse">MSE (lower is better)</option><option value="r2">R² (higher is better)</option></select></label></div>
  @if (error()) { <div class="d-alert err">{{ error() }}</div> }
  @if (cat(); as c) {
    <div class="d-alert">All six models were scored on the <b>same {{ c.test_labels.length }} held-out bills</b> ({{ c.test_labels[0] }} → {{ c.test_labels.at(-1) }}) using walk-forward testing. Unit: {{ c.unit }} ({{ c.unit_note }}). With so few test bills, small gaps between models are not conclusive. Run: {{ c.run_id }}.</div>
    <section class="d-panel"><h3>Comparison table</h3>
      <div class="d-scroll"><table class="d-table sortable"><thead><tr>
        @for (h of heads; track h.k) { <th (click)="sort(h.k)" [class.on]="sortKey() === h.k" tabindex="0" (keydown.enter)="sort(h.k)">{{ h.t }} {{ sortKey() === h.k ? (asc() ? '▲' : '▼') : '' }}</th> }</tr></thead><tbody>
        @for (m of sorted(); track m.key) { <tr [class.best]="m.key === c.best.key"><td><b>{{ m.label }}</b> @if (m.key === c.best.key) { <span class="d-chip s-passed">Best by {{ c.best.metric.toUpperCase() }}</span> } @if (m.deployed) { <span class="d-chip s-ready">Deployed</span> }<br><small class="d-muted">{{ m.description }}</small></td>
          <td>{{ m.type }}</td><td>@if (m.has_saved_weights) { <span class="d-chip s-passed">Saved weights</span> } @else if (m.type === 'baseline') { <span class="d-chip s-neutral">No training needed</span> } @else { <span class="d-chip s-warning">Evaluation only</span> }</td>
          <td>{{ fmt(m.rmse) }}</td><td>{{ fmt(m.mae) }}</td><td>{{ fmt(m.mse) }}</td><td [class.neg]="m.r2 < 0">{{ fmt(m.r2, 3) }}</td><td>{{ fmt(m.mape, 2) }}%</td><td><b>{{ fmt(m.accuracy, 2) }}%</b></td><td>{{ m.evaluation_status }}<br><small class="d-muted">{{ m.training_status }}</small></td></tr> }</tbody></table></div>
      <p class="d-muted">Evaluated {{ when(c.models[0].evaluated_at) }}. Only models marked "Saved weights" can make new predictions; the others have results only because their weights were not saved during walk-forward testing.</p></section>
    <div class="d-grid2">
      <section class="d-panel"><h3>RMSE by model</h3><p class="d-muted">Lower is better.</p><d-chart [config]="bar('rmse')" label="RMSE comparison" /></section>
      <section class="d-panel"><h3>MAE by model</h3><p class="d-muted">Lower is better.</p><d-chart [config]="bar('mae')" label="MAE comparison" /></section>
      <section class="d-panel"><h3>R² by model</h3><p class="d-muted">Higher is better. Below 0 means worse than predicting the test-period average.</p><d-chart [config]="bar('r2')" label="R squared comparison" /></section>
      <section class="d-panel"><h3>Error by forecast distance (MAE)</h3><p class="d-muted">How the error grows when forecasting 1, 2, 3 bills ahead.</p>@if (hz().length) { <d-chart [config]="hzChart()" label="Horizon MAE" /> } @else { <div class="d-empty">No horizon metrics saved.</div> }</section>
    </div>
    @if (wf(); as w) {
      <section class="d-panel"><h3>Actual versus predicted</h3>
        <label class="d-field">Model<select [ngModel]="pick()" (ngModelChange)="pick.set($event)">@for (m of c.models; track m.key) { <option [value]="m.key">{{ m.label }}</option> }</select></label>
        <d-chart [config]="avp()" [height]="320" label="Actual versus predicted" /></section>
      <section class="d-panel"><h3>Residual analysis</h3><p class="d-muted">Residual = actual − predicted. Bars above zero mean the model under-predicted.</p><d-chart [config]="resid()" label="Residuals" /></section>
    }
    <section class="d-panel"><h3>Training, validation and cross-validation</h3><p class="d-muted">Accuracy = 100 − MAPE. Learning curves, k-fold time-series cross-validation, train-vs-test accuracy and residual checks are on the <a routerLink="/training">Training and Validation</a> page.</p></section>
  } @else if (!error()) { <div class="d-loading">Loading real evaluation results…</div> }`,
})
export class Compare implements OnInit {
  private api = inject(Dash);
  metric = signal('rmse'); cat = signal<any>(null); wf = signal<any>(null); hz = signal<any[]>([]); error = signal('');
  pick = signal('cnn_lstm_attention'); sortKey = signal('rmse'); asc = signal(true); fmt = fmt; when = when;
  heads = [{ k: 'label', t: 'Model' }, { k: 'type', t: 'Type' }, { k: 'has_saved_weights', t: 'Weights' }, { k: 'rmse', t: 'RMSE' }, { k: 'mae', t: 'MAE' }, { k: 'mse', t: 'MSE' }, { k: 'r2', t: 'R²' }, { k: 'mape', t: 'MAPE' }, { k: 'accuracy', t: 'Accuracy' }, { k: 'evaluation_status', t: 'Evaluation status' }];
  sorted = computed(() => { const k = this.sortKey(), dir = this.asc() ? 1 : -1; return [...this.cat().models].sort((a: any, b: any) => (a[k] > b[k] ? 1 : a[k] < b[k] ? -1 : 0) * dir); });
  sort(k: string) { if (this.sortKey() === k) this.asc.update((v) => !v); else { this.sortKey.set(k); this.asc.set(true); } }
  ngOnInit() { this.load(); this.api.get('/models/walk-forward').subscribe((w) => this.wf.set(w)); this.api.get('/models/horizon').subscribe((h) => this.hz.set(h.rows)); }
  load() { this.api.get('/models/catalog', { metric: this.metric() }).subscribe({ next: (c) => this.cat.set(c), error: (e) => this.error.set(errText(e)) }); }
  bar(k: string): ChartConfiguration { const c = this.cat(); const ms = c.models;
    return { type: 'bar', data: { labels: ms.map((m: any) => m.label), datasets: [{ label: k.toUpperCase(), data: ms.map((m: any) => m[k]), backgroundColor: ms.map((m: any) => (m.deployed ? COLORS.blue : m.type === 'baseline' ? COLORS.grey : COLORS.sky)) }] },
      options: baseOptions('', k === 'r2' ? 'R²' : c.unit, { plugins: { legend: { display: false } }, scales: { x: { ticks: { maxRotation: 40 }, grid: { display: false } }, y: { title: { display: true, text: k === 'r2' ? 'R²' : c.unit } } } }) }; }
  hzChart = computed<ChartConfiguration>(() => { const rows = this.hz(); const models = [...new Set(rows.map((r) => r.model))]; const hs = [...new Set(rows.map((r) => r.horizon))].sort();
    return { type: 'line', data: { labels: hs.map((h) => `${h} ahead`), datasets: models.map((m, i) => ({ label: m as string, borderColor: PALETTE[i % 6], data: hs.map((h) => rows.find((r) => r.model === m && r.horizon === h)?.mae ?? null) })) }, options: baseOptions('Forecast distance (bills ahead)', 'MAE') }; });
  avp = computed<ChartConfiguration>(() => { const w = this.wf(); return { type: 'line', data: { labels: w.labels, datasets: [{ label: 'Actual', data: w.actual, borderColor: COLORS.blue }, { label: 'Predicted', data: w.predictions[this.pick()], borderColor: COLORS.amber, borderDash: [5, 4] }] }, options: baseOptions('Test billing period', w.unit) }; });
  resid = computed<ChartConfiguration>(() => { const w = this.wf(); return { type: 'bar', data: { labels: w.labels, datasets: [{ label: 'Residual', data: w.residuals[this.pick()], backgroundColor: w.residuals[this.pick()].map((v: number) => (v >= 0 ? COLORS.sky : COLORS.red)) }] }, options: baseOptions('Test billing period', 'Actual − predicted', { plugins: { legend: { display: false } } }) }; });
}

// ---------------------------------------------------------------- Training / testing / validation
@Component({
  selector: 'd-training',
  standalone: true,
  imports: [DChart, StatusChip],
  template: `
  <div class="d-pagehead"><h2>Training, Testing and Validation</h2></div>
  @if (error()) { <div class="d-alert err">{{ error() }}</div> }
  @if (t(); as t) {
    <div class="d-alert"><b>Accuracy</b> = 100 − MAPE (mean absolute percentage error). Forecasting is a regression task, so this is "how close, on average, the forecast is to the actual bill", not a classification hit rate.</div>
    <section class="d-panel"><h3>A. Training results <small class="d-muted">(deployed model: {{ t.training.deployed_variant }})</small></h3>
      <div class="d-stats"><div><b>{{ t.training.training_periods }}</b> billing periods used</div><div><b>{{ t.training.training_windows }}</b> training windows</div><div><b>{{ t.training.final_training_epochs ?? '—' }}</b> final epochs</div><div>window <b>{{ t.training.sequence_length }}</b> bills</div></div>
      <table class="d-table"><thead><tr><th>In-sample fit (saved weights)</th><th>Training accuracy</th><th>RMSE</th><th>MAE</th><th>MSE</th><th>R²</th></tr></thead><tbody>
        <tr><td>One-step, {{ t.training.in_sample_metrics.n }} windows</td><td><b>{{ pct(t.training.in_sample_metrics.accuracy) }}</b></td><td>{{ fmt(t.training.in_sample_metrics.rmse) }}</td><td>{{ fmt(t.training.in_sample_metrics.mae) }}</td><td>{{ fmt(t.training.in_sample_metrics.mse) }}</td><td>{{ fmt(t.training.in_sample_metrics.r2, 3) }}</td></tr></tbody></table>
      <p class="d-muted">{{ t.training.in_sample_note }} {{ t.training.duration_note }}</p>
      @if (t.loss_history.recorded) {
        <h4>Learning curve (final model)</h4><p class="d-muted">{{ t.loss_history.message }} Best epoch: <b>{{ t.loss_history.best_epoch }}</b>. Loss unit: {{ t.loss_history.unit }}.</p>
        <d-chart [config]="lossChart()" label="Training and validation loss per epoch" />
      } @else { <div class="d-alert warn"><b>Loss curves unavailable.</b> {{ t.loss_history.message }}</div> }
      <details><summary>Hyperparameters of the deployed model</summary><div class="d-scroll"><table class="d-table"><tbody>@for (p of params(); track p[0]) { <tr><td>{{ p[0] }}</td><td>{{ p[1] }}</td></tr> }</tbody></table></div></details></section>

    <section class="d-panel"><h3>B. Testing results</h3><p class="d-muted">{{ t.testing.protocol }} — {{ t.testing.n_test_points }} held-out test bills.</p>
      <div class="d-scroll"><table class="d-table"><thead><tr><th>Model</th><th>Test accuracy</th><th>RMSE</th><th>MAE</th><th>MSE</th><th>R²</th><th>MAPE</th></tr></thead><tbody>
        @for (m of t.testing.models; track m.key) { <tr><td>{{ m.label }}</td><td><b>{{ pct(m.accuracy) }}</b></td><td>{{ fmt(m.rmse) }}</td><td>{{ fmt(m.mae) }}</td><td>{{ fmt(m.mse) }}</td><td [class.neg]="m.r2 < 0">{{ fmt(m.r2, 3) }}</td><td>{{ fmt(m.mape, 2) }}%</td></tr> }</tbody></table></div>
      <p class="d-muted">Actual-versus-predicted and residual plots are on the Model Comparison page.</p></section>

    <section class="d-panel"><h3>C. Training vs testing accuracy</h3>
      @if (t.cross_validation.recorded) {
        <p class="d-muted">Train, validation and CV-test accuracy are averages over the {{ t.cross_validation.n_folds }} cross-validation folds; walk-forward test is the main held-out evaluation. A model is flagged when its training accuracy is more than 5 points above its test accuracy (a sign of overfitting).</p>
        <div class="d-scroll"><table class="d-table"><thead><tr><th>Model</th><th>Train</th><th>Validation</th><th>CV test</th><th>Walk-forward test</th><th>Train − test gap</th><th>Check</th></tr></thead><tbody>
          @for (a of t.cross_validation.accuracy; track a.model) { <tr><td>{{ a.model }}</td><td>{{ pct(a.train_accuracy) }}</td><td>{{ pct(a.validation_accuracy) }}</td><td><b>{{ pct(a.cv_test_accuracy) }}</b></td><td>{{ pct(a.walk_forward_test_accuracy) }}</td><td>{{ a.train_test_gap === null ? '—' : fmt(a.train_test_gap, 2) + ' pts' }}</td>
            <td>@if (a.fit_check === 'ok') { <span class="d-chip s-passed">ok</span> } @else if (a.fit_check === 'possible overfitting') { <span class="d-chip s-warning">possible overfitting</span> } @else { <span class="d-chip s-neutral">{{ a.fit_check }}</span> }</td></tr> }</tbody></table></div>
        <d-chart [config]="accChart()" [height]="320" label="Train, validation and test accuracy by model" />
      } @else { <div class="d-alert warn">This model run did not record train/validation/test accuracy. Re-run <code>run_kharghar_experiment.py</code> to produce it.</div> }</section>

    <section class="d-panel"><h3>D. Validation of the model</h3>
      <p><b>Early stopping.</b> {{ t.validation.method }}</p><p><b>Cross-validation.</b> {{ t.validation.cross_validation }}</p>
      @if (t.cross_validation.recorded) {
        <h4>{{ t.cross_validation.n_folds }}-fold time-series cross-validation</h4><p class="d-muted">{{ t.cross_validation.method }}</p>
        <div class="d-scroll"><table class="d-table"><thead><tr><th>Fold</th><th>Trained on</th><th>Tested on</th></tr></thead><tbody>
          @for (f of t.cross_validation.folds; track f.fold) { <tr><td>{{ f.fold }}</td><td>first {{ f.train_bills }} bills</td><td>{{ f.test_from }} → {{ f.test_to }} ({{ f.test_bills }} bills)</td></tr> }</tbody></table></div>
        <h4>Mean ± standard deviation across folds</h4>
        <div class="d-scroll"><table class="d-table"><thead><tr><th>Model</th><th>Split</th><th>Accuracy</th><th>RMSE</th><th>MAE</th><th>R²</th></tr></thead><tbody>
          @for (r of t.cross_validation.summary; track r.model + r.split) { <tr><td>{{ r.model }}</td><td>{{ r.split }}</td><td><b>{{ pct(r.accuracy_pct_mean) }}</b> ± {{ fmt(r.accuracy_pct_std, 2) }}</td><td>{{ fmt(r.rmse_mean) }} ± {{ fmt(r.rmse_std) }}</td><td>{{ fmt(r.mae_mean) }} ± {{ fmt(r.mae_std) }}</td><td [class.neg]="r.r2_mean < 0">{{ fmt(r.r2_mean, 3) }}</td></tr> }</tbody></table></div>
        <p class="d-muted">Each test block has only a few bills, so R² per fold is unstable; accuracy and RMSE are more reliable here.</p>
        <div class="d-grid2">
          <section><h4>Test accuracy per fold</h4><d-chart [config]="foldChart()" label="Test accuracy per cross-validation fold" /></section>
          @if (t.loss_history.recorded) { <section><h4>Validation loss per fold ({{ t.training.deployed_variant }})</h4><d-chart [config]="foldLossChart()" label="Validation loss per fold" /></section> }
        </div>
      } @else { <div class="d-alert warn">No k-fold cross-validation was recorded for this run.</div> }
      @if (t.residual_diagnostics.length) {
        <h4>Residual analysis (walk-forward test bills)</h4>
        <p class="d-muted">Residual = actual − forecast. Bias t-statistic beyond ±2 means the model systematically over- or under-predicts; lag-1 autocorrelation near 0 means errors do not follow a pattern.</p>
        <div class="d-scroll"><table class="d-table"><thead><tr><th>Model</th><th>Mean residual</th><th>Std</th><th>Largest miss</th><th>Bias t</th><th>Lag-1 autocorr.</th><th>Under-predicted</th><th>Bias check</th></tr></thead><tbody>
          @for (r of t.residual_diagnostics; track r.model) { <tr><td>{{ r.model }}</td><td>{{ fmt(r.mean_residual) }}</td><td>{{ fmt(r.residual_std) }}</td><td>{{ fmt(r.max_abs_residual) }}</td><td>{{ fmt(r.bias_t, 2) }}</td><td>{{ fmt(r.lag1_autocorr, 2) }}</td><td>{{ fmt(r.share_under_predicted * 100, 0) }}%</td>
            <td><span class="d-chip" [class.s-passed]="r.bias_check === 'ok'" [class.s-warning]="r.bias_check === 'biased'">{{ r.bias_check }}</span></td></tr> }</tbody></table></div>
      }
      @if (t.validation.tuning_trials.length) { <h4>Hyperparameter tuning score per trial</h4><p class="d-muted">{{ t.validation.tuning_note }}</p><d-chart [config]="trials()" label="Tuning trials" /> }</section>
  }
  @if (v(); as v) {
    <section class="d-panel"><h3>Model validation and reliability <d-status [status]="v.overall" /></h3>
      <p class="d-muted">Each status comes from a check that actually ran. Passed = check ran and passed · Warning = ran, needs caution · Failed = ran and failed · Not evaluated = not possible with the saved artifacts.</p>
      <table class="d-table"><thead><tr><th>Check</th><th>Status</th><th>Reason</th></tr></thead><tbody>
        @for (c of v.checks; track c.name) { <tr><td>{{ c.name }}</td><td><d-status [status]="c.status" /></td><td>{{ c.detail }}</td></tr> }</tbody></table></section>
  } @else if (!error()) { <div class="d-loading">Loading…</div> }`,
})
export class Training implements OnInit {
  private api = inject(Dash);
  t = signal<any>(null); v = signal<any>(null); error = signal(''); fmt = fmt;
  pct = (x: number | null | undefined) => (x === null || x === undefined ? '—' : `${fmt(x, 2)}%`);
  params = computed(() => Object.entries(this.t().training.hyperparameters).map(([k, x]) => [k, typeof x === 'number' ? fmt(x, 5) : String(x)]));
  trials = computed<ChartConfiguration>(() => { const rows = this.t().validation.tuning_trials; const key = Object.keys(rows[0]).find((k) => ['value', 'score', 'val_loss'].includes(k)) ?? Object.keys(rows[0]).at(-1)!;
    return { type: 'bar', data: { labels: rows.map((_: any, i: number) => `#${i + 1}`), datasets: [{ label: key, data: rows.map((r: any) => r[key]), backgroundColor: COLORS.sky }] }, options: baseOptions('Trial', key, { plugins: { legend: { display: false } } }) }; });
  lossChart = computed<ChartConfiguration>(() => { const f = this.t().loss_history.final;
    return { type: 'line', data: { labels: f.epoch, datasets: [{ label: 'Training loss', data: f.train_loss, borderColor: COLORS.blue, pointRadius: 0 }, { label: 'Validation loss', data: f.val_loss, borderColor: COLORS.amber, pointRadius: 0 }] }, options: baseOptions('Epoch', 'MSE (scaled)') }; });
  accChart = computed<ChartConfiguration>(() => { const rows = this.t().cross_validation.accuracy.filter((a: any) => a.type === 'neural');
    const sets: [string, string, string][] = [['train_accuracy', 'Train', COLORS.blue], ['validation_accuracy', 'Validation', COLORS.amber], ['cv_test_accuracy', 'CV test', COLORS.green], ['walk_forward_test_accuracy', 'Walk-forward test', COLORS.grey]];
    return { type: 'bar', data: { labels: rows.map((a: any) => a.model), datasets: sets.map(([k, label, color]) => ({ label, data: rows.map((a: any) => a[k]), backgroundColor: color })) }, options: baseOptions('Model', 'Accuracy (%)', { scales: { x: { grid: { display: false } }, y: { title: { display: true, text: 'Accuracy (%)' }, suggestedMax: 100 } } }) }; });
  foldChart = computed<ChartConfiguration>(() => { const cv = this.t().cross_validation; const test = cv.per_fold.filter((r: any) => r.split === 'test');
    const models = [...new Set(test.map((r: any) => r.model))] as string[]; const folds = cv.folds.map((f: any) => f.fold);
    return { type: 'bar', data: { labels: folds.map((f: number) => `Fold ${f}`), datasets: models.map((m, i) => ({ label: m, backgroundColor: PALETTE[i % PALETTE.length], data: folds.map((f: number) => test.find((r: any) => r.model === m && r.fold === f)?.accuracy_pct ?? null) })) }, options: baseOptions('Fold', 'Test accuracy (%)', { scales: { x: { grid: { display: false } }, y: { title: { display: true, text: 'Test accuracy (%)' }, suggestedMax: 100 } } }) }; });
  foldLossChart = computed<ChartConfiguration>(() => { const folds = this.t().loss_history.folds; const keys = Object.keys(folds); const maxLen = Math.max(...keys.map((k) => folds[k].epoch.length));
    return { type: 'line', data: { labels: Array.from({ length: maxLen }, (_, i) => i + 1), datasets: keys.map((k, i) => ({ label: `Fold ${k}`, data: folds[k].val_loss, borderColor: PALETTE[i % PALETTE.length], pointRadius: 0 })) }, options: baseOptions('Epoch', 'Validation MSE (scaled)') }; });
  ngOnInit() {
    this.api.get('/models/training').subscribe({ next: (r) => this.t.set(r), error: (e) => this.error.set(errText(e)) });
    this.api.get('/models/validation').subscribe({ next: (r) => this.v.set(r), error: (e) => this.error.set(errText(e)) });
  }
}

// ---------------------------------------------------------------- Confusion matrix
@Component({
  selector: 'd-confusion',
  standalone: true,
  imports: [FormsModule],
  template: `
  <div class="d-pagehead"><h2>Confusion Matrix</h2></div>
  <div class="d-alert warn"><b>Derived evaluation, not a native classifier.</b> None of the project's models is a classifier. Actual and predicted consumption are mapped to low / normal / high with the same thresholds, and the matrix compares those categories. It is not the models' own classification accuracy.</div>
  <div class="d-filters">
    <label class="d-field">Model<select [(ngModel)]="model" (change)="load()">@for (m of models; track m[0]) { <option [value]="m[0]">{{ m[1] }}</option> }</select></label>
    <label class="d-field">Low below quantile<input type="number" step="0.05" min="0.05" max="0.9" [(ngModel)]="qLow" (change)="load()"></label>
    <label class="d-field">High above quantile<input type="number" step="0.05" min="0.1" max="0.95" [(ngModel)]="qHigh" (change)="load()"></label></div>
  @if (error()) { <div class="d-alert err">{{ error() }}</div> }
  @if (c(); as c) {
    <p class="d-muted">{{ c.disclaimer }} Thresholds: low &lt; <b>{{ fmt(c.thresholds.low_below) }}</b>, high &gt; <b>{{ fmt(c.thresholds.high_above) }}</b> (from the {{ c.thresholds.fit_on }}).</p>
    @for (w of c.warnings; track w) { <div class="d-alert warn">{{ w }}</div> }
    <div class="d-grid2">
      <section class="d-panel"><h3>{{ c.model_label }} — {{ c.n }} test bills</h3>
        <div class="d-cm" role="table" aria-label="Confusion matrix">
          <div class="d-cm-corner">Actual ↓ / Predicted →</div>@for (k of c.classes; track k) { <div class="d-cm-head">{{ k }}</div> }
          @for (row of c.matrix; track $index; let i = $index) { <div class="d-cm-head">{{ c.classes[i] }}</div>
            @for (v of row; track $index; let j = $index) { <div class="d-cm-cell" [class.diag]="i === j" [style.--a]="c.n ? v / c.n : 0">{{ v }}</div> } }
        </div>
        <div class="d-stats"><div>Correct <b>{{ c.correct }}</b></div><div>Incorrect <b>{{ c.incorrect }}</b></div><div>Category accuracy <b>{{ c.accuracy !== null ? fmt(c.accuracy * 100, 1) + '%' : '—' }}</b></div><div>Always-predict-most-common-class baseline <b>{{ fmt(c.majority_class_accuracy * 100, 1) }}%</b></div></div>
        @if (!c.accuracy_informative) { <div class="d-alert warn">This accuracy is <b>not informative</b>: only one class occurs among the test bills.</div> }
        <p class="d-muted">Accuracy = sum of the diagonal ÷ all evaluated bills.</p></section>
      <section class="d-panel"><h3>Per-class metrics</h3>
        <table class="d-table"><thead><tr><th>Class</th><th>Support</th><th>Precision</th><th>Recall</th><th>F1</th></tr></thead><tbody>
          @for (p of c.per_class; track p.class) { <tr><td>{{ p.class }}</td><td>{{ p.support }}</td><td>{{ p.precision !== null ? fmt(p.precision, 2) : 'n/a' }}</td><td>{{ p.recall !== null ? fmt(p.recall, 2) : 'n/a' }}</td><td>{{ p.f1 !== null ? fmt(p.f1, 2) : 'n/a' }}</td></tr> }</tbody></table>
        <h4>Regression metrics for the same bills</h4>
        <div class="d-stats"><div>RMSE <b>{{ fmt(c.regression_metrics.rmse) }}</b></div><div>MAE <b>{{ fmt(c.regression_metrics.mae) }}</b></div><div>MSE <b>{{ fmt(c.regression_metrics.mse) }}</b></div><div>R² <b>{{ fmt(c.regression_metrics.r2, 3) }}</b></div></div></section>
    </div>
    <section class="d-panel"><h3>Bill-by-bill categories</h3><div class="d-scroll"><table class="d-table"><thead><tr><th>Bill</th><th>Actual category</th><th>Predicted category</th><th>Match</th></tr></thead><tbody>
      @for (l of c.labels; track l; let i = $index) { <tr><td>{{ l }}</td><td>{{ c.actual_classes[i] }}</td><td>{{ c.predicted_classes[i] }}</td><td>{{ c.actual_classes[i] === c.predicted_classes[i] ? '✓' : '✗' }}</td></tr> }</tbody></table></div></section>
  } @else if (!error()) { <div class="d-loading">Loading…</div> }`,
})
export class Confusion implements OnInit {
  private api = inject(Dash);
  c = signal<any>(null); error = signal(''); model = 'cnn_lstm_attention'; qLow = 0.33; qHigh = 0.67; fmt = fmt;
  models = [['cnn_lstm_attention', 'CNN-LSTM + Attention'], ['cnn_lstm', 'CNN-LSTM'], ['lstm_attention', 'LSTM + Attention'], ['lstm', 'LSTM'], ['naive_last_value', 'Repeat last value'], ['seasonal_naive_6', 'Same bill last year']];
  ngOnInit() { this.load(); }
  load() { this.error.set(''); this.api.get('/models/confusion', { model: this.model, q_low: this.qLow, q_high: this.qHigh }).subscribe({ next: (r) => this.c.set(r), error: (e) => this.error.set(errText(e)) }); }
}
