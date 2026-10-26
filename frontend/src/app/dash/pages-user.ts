import { Component, OnInit, computed, inject, input, model, signal } from '@angular/core';
import { RouterLink } from '@angular/router';
import { FormsModule } from '@angular/forms';
import { ChartConfiguration } from 'chart.js';
import { COLORS, DChart, Dash, HttpEventType, Auth, Toasts, baseOptions, errText, fmt, when } from './core';

// ---------------------------------------------------------------- shared pieces
@Component({
  selector: 'd-dataset-pick',
  standalone: true,
  imports: [FormsModule],
  template: `<label class="d-field">Dataset
    <select [ngModel]="value()" (ngModelChange)="value.set(+$event)" aria-label="Dataset">
      @for (d of options(); track d.id) { <option [value]="d.id">{{ d.id === 0 ? 'Original project dataset' : d.name + (d.status !== 'ready' ? ' (unconfirmed)' : '') }}</option> }
    </select></label>`,
})
export class DatasetPick implements OnInit {
  private api = inject(Dash);
  value = model(0);
  readyOnly = input(false);
  options = signal<any[]>([{ id: 0, status: 'ready' }]);
  ngOnInit() {
    this.api.get('/datasets').subscribe((r) => this.options.set([r.original, ...r.datasets.filter((d: any) => !this.readyOnly() || d.status === 'ready').filter((d: any) => d.status !== 'rejected')]));
  }
}

@Component({
  selector: 'd-status',
  standalone: true,
  template: `<span class="d-chip" [class]="'s-' + status()">{{ label() }}</span>`,
})
export class StatusChip {
  status = model('passed');
  label = computed(() => ({ passed: 'Passed', warning: 'Warning', failed: 'Failed', not_evaluated: 'Not evaluated', ready: 'Ready', pending_confirmation: 'Awaiting confirmation', rejected: 'Rejected', deleted: 'Deleted', ok: 'OK', denied: 'Denied' } as any)[this.status()] ?? this.status());
}

const lineCfg = (labels: string[], sets: any[], x: string, y: string): ChartConfiguration => ({ type: 'line', data: { labels, datasets: sets }, options: baseOptions(x, y) });

// ---------------------------------------------------------------- Overview
@Component({
  selector: 'd-overview',
  standalone: true,
  imports: [DatasetPick, DChart, StatusChip, RouterLink],
  template: `
  <div class="d-pagehead"><h2>Overview</h2><d-dataset-pick [(value)]="datasetId" (valueChange)="load()" /></div>
  @if (loading()) { <div class="d-loading">Loading real results…</div> }
  @if (error()) { <div class="d-alert err">{{ error() }}</div> }
  @if (d(); as o) {
    <p class="d-muted">Unit: <b>{{ o.unit }}</b> — {{ o.unit_note }}. Dataset: {{ o.dataset.name }}. Model run: {{ o.run_id }}.</p>
    <section class="d-cards">
      <div class="d-card"><span>Total dataset records</span><b>{{ o.stats.records }}</b><small>{{ o.stats.first_label }} → {{ o.stats.latest_label }}</small></div>
      <div class="d-card"><span>Total historical consumption</span><b>{{ f(o.stats.total) }}</b><small>{{ o.unit }}</small></div>
      <div class="d-card"><span>Average per billing period</span><b>{{ f(o.stats.average) }}</b><small>{{ o.unit }}</small></div>
      <div class="d-card"><span>Maximum observed</span><b>{{ f(o.stats.max) }}</b><small>{{ o.stats.max_label }}</small></div>
      <div class="d-card"><span>Minimum observed</span><b>{{ f(o.stats.min) }}</b><small>{{ o.stats.min_label }}</small></div>
      <div class="d-card"><span>Latest recorded</span><b>{{ f(o.stats.latest) }}</b><small>{{ o.stats.latest_label }}</small></div>
      <div class="d-card hl"><span>Forecast: next billing period</span>
        @if (o.forecast_next) { <b>{{ f(o.forecast_next.forecast) }}</b><small>{{ o.forecast_next.label }}@if (o.forecast_next.lower !== null) { · ±{{ f(o.forecast_next.forecast - o.forecast_next.lower) }} typical error }</small> }
        @else { <b>—</b><small>Needs at least 6 periods.</small> }</div>
      <div class="d-card hl"><span>Best model (lowest RMSE)</span><b class="sm">{{ o.best_model.label }}</b><small>{{ o.best_model.basis }}</small></div>
      <div class="d-card"><span>Best model RMSE</span><b>{{ f(o.best_model.rmse) }}</b><small>lower is better</small></div>
      <div class="d-card"><span>Best model MAE</span><b>{{ f(o.best_model.mae) }}</b><small>lower is better</small></div>
      <div class="d-card"><span>Best model R²</span><b [class.neg]="o.best_model.r2 < 0">{{ fmt(o.best_model.r2, 3) }}</b><small>{{ o.best_model.r2 < 0 ? 'negative: worse than predicting the test mean' : 'higher is better' }}</small></div>
      <div class="d-card"><span>Your last upload</span><b class="sm">{{ when(o.dataset_last_uploaded) }}</b><small>{{ o.dataset_last_uploaded ? '' : 'No upload yet' }}</small></div>
      <div class="d-card"><span>Model validation status</span><b><d-status [status]="o.validation_status" /></b><small><a routerLink="/training">See the checks</a></small></div>
    </section>
    <section class="d-panel"><h3>Billing-period history and next forecast</h3><d-chart [config]="chart()!" [height]="320" label="History and forecast" /></section>
  } @else if (!loading() && !error()) { <div class="d-empty">No data loaded.</div> }`,
})
export class Overview implements OnInit {
  private api = inject(Dash);
  datasetId = signal(0);
  d = signal<any>(null); loading = signal(false); error = signal('');
  f = (n: number) => fmt(n); fmt = fmt; when = when;
  chart = computed<ChartConfiguration | null>(() => {
    const o = this.d(); if (!o) return null;
    const labels = [...o.history.labels, ...(o.forecast_next ? [o.forecast_next.label] : [])];
    const hist = [...o.history.values, ...(o.forecast_next ? [null] : [])];
    const fc = o.history.values.map(() => null as number | null);
    if (o.forecast_next) { fc[fc.length - 1] = o.history.values.at(-1); fc.push(o.forecast_next.forecast); }
    return lineCfg(labels, [
      { label: 'Actual', data: hist, borderColor: COLORS.blue, backgroundColor: 'rgba(21,101,192,.1)', fill: true, tension: .2 },
      { label: 'Forecast', data: fc, borderColor: COLORS.amber, borderDash: [6, 4], pointRadius: 5 }], 'Billing period', o.unit);
  });
  ngOnInit() { this.load(); }
  load() {
    this.loading.set(true); this.error.set('');
    this.api.get('/dashboard/overview', { dataset_id: this.datasetId() }).subscribe({
      next: (r) => { this.d.set(r); this.loading.set(false); }, error: (e) => { this.error.set(errText(e)); this.loading.set(false); } });
  }
}

// ---------------------------------------------------------------- Upload
@Component({
  selector: 'd-upload',
  standalone: true,
  imports: [StatusChip],
  template: `
  <div class="d-pagehead"><h2>Upload Dataset</h2></div>
  <section class="d-panel">
    <div class="d-drop" (dragover)="$event.preventDefault()" (drop)="drop($event)">
      <input #fi type="file" accept=".csv,.xlsx" (change)="pick($any($event.target).files)" hidden>
      @if (file(); as f) {
        <div><b>{{ f.name }}</b> · {{ (f.size / 1024).toFixed(1) }} KB</div>
        <div class="d-row">
          <button class="d-btn primary" (click)="send()" [disabled]="busy()">Upload and validate</button>
          <button class="d-btn" (click)="fi.click()" [disabled]="busy()">Replace file</button>
          <button class="d-btn" (click)="reset()" [disabled]="busy()">Cancel</button>
        </div>
        @if (busy()) { <div class="d-progress"><i [style.width.%]="progress()"></i></div><small>{{ progress() < 100 ? 'Uploading… ' + progress() + '%' : 'Validating…' }}</small> }
      } @else {
        <p>Drag a file here or</p><button class="d-btn primary" (click)="fi.click()">Browse files</button>
      }
    </div>
  </section>
  @if (error()) { <div class="d-alert err" role="alert">{{ error() }}</div> }
  @if (result(); as r) {
    <section class="d-panel">
      <h3>Validation result <d-status [status]="r.validation.status" /></h3>
      <div class="d-stats">
        <div><b>{{ r.validation.n_rows ?? '—' }}</b> rows</div><div><b>{{ r.validation.n_cols ?? '—' }}</b> columns</div>
        <div><b>{{ r.validation.missing_values }}</b> missing values</div><div><b>{{ r.validation.duplicate_records }}</b> duplicate rows</div>
        <div><b>{{ r.validation.invalid_dates }}</b> invalid dates</div><div><b>{{ r.validation.invalid_values }}</b> invalid values</div>
      </div>
      @if (r.validation.errors.length) { <div class="d-alert err"><b>Why this file was rejected</b><ul>@for (e of r.validation.errors; track e) { <li>{{ e }}</li> }</ul></div> }
      @if (r.validation.warnings.length) { <div class="d-alert warn"><b>Warnings</b><ul>@for (w of r.validation.warnings; track w) { <li>{{ w }}</li> }</ul></div> }
      <table class="d-table"><thead><tr><th>Check</th><th>Status</th><th>Detail</th></tr></thead><tbody>
        @for (c of r.validation.checks; track c.name) { <tr><td>{{ c.name }}</td><td><d-status [status]="c.status" /></td><td>{{ c.detail }}</td></tr> }</tbody></table>
      @if (r.validation.columns.length) {
        <h4>Columns and detected types</h4>
        <table class="d-table"><thead><tr><th>Column</th><th>Type</th><th>Missing</th></tr></thead><tbody>
          @for (c of r.validation.columns; track c.name) { <tr><td>{{ c.name }}</td><td>{{ c.dtype }}</td><td>{{ c.missing }}</td></tr> }</tbody></table>
      }
      @if (r.validation.preview.length) {
        <h4>Preview (first {{ r.validation.preview.length }} rows)</h4>
        <div class="d-scroll"><table class="d-table"><thead><tr>@for (k of keys(r.validation.preview[0]); track k) { <th>{{ k }}</th> }</tr></thead><tbody>
          @for (row of r.validation.preview; track $index) { <tr>@for (k of keys(row); track k) { <td>{{ row[k] }}</td> }</tr> }</tbody></table></div>
      }
      <div class="d-row">
        @if (r.dataset_id) { <button class="d-btn" (click)="report(r.dataset_id)">Download validation report</button> }
        @if (r.status === 'pending_confirmation') {
          @if (!confirming()) { <button class="d-btn primary" (click)="confirming.set(true)">Use this dataset…</button> }
          @else { <span class="d-confirm">Confirm that you reviewed the validation above? <button class="d-btn primary" (click)="confirm(r.dataset_id)">Yes, confirm</button> <button class="d-btn" (click)="confirming.set(false)">No</button></span> }
        }
        @if (r.status === 'ready') { <span class="d-chip s-passed">Confirmed — ready for prediction</span> }
      </div>
    </section>
  }
  <section class="d-panel"><h3>Upload history</h3>
    @if (history().length === 0) { <div class="d-empty">No uploads yet.</div> } @else {
    <div class="d-scroll"><table class="d-table"><thead><tr><th>File</th><th>Uploaded</th><th>Rows</th><th>Size</th><th>Validation</th><th>Status</th><th></th></tr></thead><tbody>
      @for (h of history(); track h.id) { <tr><td>{{ h.name }}</td><td>{{ when(h.uploaded_at) }}</td><td>{{ h.n_rows ?? '—' }}</td><td>{{ (h.size_bytes / 1024).toFixed(1) }} KB</td>
        <td><d-status [status]="h.validation_status" /></td><td><d-status [status]="h.status" /></td>
        <td class="d-actions">@if (h.status === 'pending_confirmation') { <button class="d-btn sm" (click)="confirm(h.id)">Confirm</button> }
          @if (h.status !== 'rejected') { <button class="d-btn sm" (click)="view(h.id)">Preview</button> }
          <button class="d-btn sm" (click)="report(h.id)">Report</button>
          @if (h.status !== 'rejected') { @if (delId() === h.id) { <button class="d-btn sm danger" (click)="remove(h.id)">Really delete</button> <button class="d-btn sm" (click)="delId.set(0)">Keep</button> } @else { <button class="d-btn sm danger" (click)="delId.set(h.id)">Delete</button> } }</td></tr> }
    </tbody></table></div> }
  </section>
  @if (previewRows(); as p) { <section class="d-panel"><h3>Preview: {{ p.dataset.name }} ({{ p.total_rows }} rows)</h3>
    <div class="d-scroll"><table class="d-table"><thead><tr>@for (k of p.columns; track k) { <th>{{ k }}</th> }</tr></thead><tbody>
      @for (row of p.rows; track $index) { <tr>@for (k of p.columns; track k) { <td>{{ row[k] }}</td> }</tr> }</tbody></table></div></section> }`,
})
export class Upload implements OnInit {
  private api = inject(Dash); private toast = inject(Toasts);
  file = signal<File | null>(null); busy = signal(false); progress = signal(0); error = signal('');
  result = signal<any>(null); history = signal<any[]>([]); confirming = signal(false); delId = signal(0); previewRows = signal<any>(null);
  when = when; keys = (o: any) => Object.keys(o);
  ngOnInit() { this.loadHistory(); }
  loadHistory() { this.api.get('/datasets').subscribe((r) => this.history.set(r.datasets)); }
  pick(files: FileList | null) { if (files?.length) { this.file.set(files[0]); this.result.set(null); this.error.set(''); this.confirming.set(false); } }
  drop(e: DragEvent) { e.preventDefault(); this.pick(e.dataTransfer?.files ?? null); }
  reset() { this.file.set(null); this.result.set(null); this.error.set(''); }
  send() {
    const f = this.file(); if (!f) return;
    this.busy.set(true); this.progress.set(0); this.error.set(''); this.result.set(null);
    this.api.upload(f).subscribe({
      next: (ev) => {
        if (ev.type === HttpEventType.UploadProgress && ev.total) this.progress.set(Math.round((100 * ev.loaded) / ev.total));
        if (ev.type === HttpEventType.Response) { this.busy.set(false); this.result.set(ev.body); this.toast.show('File passed validation. Review it, then confirm.'); this.loadHistory(); }
      },
      error: (e) => {
        this.busy.set(false);
        const v = e.error?.validation;
        if (v) { this.result.set({ dataset_id: e.error.dataset_id, status: 'rejected', validation: v }); this.toast.show('Dataset rejected — see the reasons below.', 'err'); this.loadHistory(); }
        else this.error.set(errText(e));
      },
    });
  }
  confirm(id: number) {
    this.api.post(`/datasets/${id}/confirm`).subscribe({
      next: () => { this.toast.show('Dataset confirmed. You can now run predictions on it.'); this.confirming.set(false); this.result.update((r) => (r && r.dataset_id === id ? { ...r, status: 'ready' } : r)); this.loadHistory(); },
      error: (e) => this.toast.show(errText(e), 'err') });
  }
  view(id: number) { this.api.get(`/datasets/${id}/preview`, { limit: 25 }).subscribe({ next: (r) => this.previewRows.set(r), error: (e) => this.toast.show(errText(e), 'err') }); }
  report(id: number) { this.api.download(`/datasets/${id}/validation-report.csv`, `validation_${id}.csv`, (m) => this.toast.show(m, 'err')); }
  remove(id: number) { this.api.del(`/datasets/${id}`).subscribe({ next: () => { this.delId.set(0); this.toast.show('Dataset deleted.'); this.previewRows.set(null); this.loadHistory(); }, error: (e) => this.toast.show(errText(e), 'err') }); }
}

// ---------------------------------------------------------------- Analytics
@Component({
  selector: 'd-analytics',
  standalone: true,
  imports: [DatasetPick, DChart, FormsModule],
  template: `
  <div class="d-pagehead"><h2>Water Consumption Analytics</h2></div>
  <div class="d-filters"><d-dataset-pick [(value)]="datasetId" (valueChange)="load()" />
    <label class="d-field">From<input type="date" [(ngModel)]="start" (change)="load()"></label>
    <label class="d-field">To<input type="date" [(ngModel)]="end" (change)="load()"></label>
    <button class="d-btn" (click)="start = ''; end = ''; load()">Clear dates</button></div>
  <p class="d-muted">The data holds one combined total per billing period, so there is no region filter.</p>
  @if (error()) { <div class="d-alert err">{{ error() }}</div> }
  @if (a(); as a) {
    @if (a.empty) { <div class="d-empty">{{ a.message }}</div> } @else {
      <section class="d-panel"><h3>Consumption per billing period</h3><d-chart [config]="trend()" [height]="320" label="Consumption trend" /></section>
      <div class="d-grid2">
        <section class="d-panel"><h3>Change from previous period (%)</h3><d-chart [config]="change()" label="Percent change" /></section>
        <section class="d-panel"><h3>Same season, year by year</h3><d-chart [config]="season()" label="Seasonal comparison" /></section>
      </div>
      <section class="d-panel"><h3>Yearly totals</h3><table class="d-table"><thead><tr><th>Year (of period end)</th><th>Periods</th><th>Total {{ a.unit }}</th></tr></thead><tbody>
        @for (y of a.yearly; track y.year) { <tr><td>{{ y.year }}</td><td>{{ y.periods }}</td><td>{{ f(y.total) }}</td></tr> }</tbody></table></section>
    }
  } @else if (!error()) { <div class="d-loading">Loading…</div> }`,
})
export class Analytics implements OnInit {
  private api = inject(Dash);
  datasetId = signal(0); a = signal<any>(null); error = signal(''); start = ''; end = ''; f = (n: number) => fmt(n);
  trend = computed(() => { const a = this.a(); return lineCfg(a.labels, [
    { label: 'Consumption', data: a.values, borderColor: COLORS.blue, backgroundColor: 'rgba(21,101,192,.1)', fill: true, tension: .2 },
    { label: '3-period moving average', data: a.rolling_mean_3, borderColor: COLORS.teal, borderDash: [5, 4], pointRadius: 0 }], 'Billing period', a.unit); });
  change = computed<ChartConfiguration>(() => { const a = this.a(); return { type: 'bar', data: { labels: a.labels, datasets: [{ label: '% change', data: a.pct_change, backgroundColor: a.pct_change.map((v: number | null) => (v ?? 0) >= 0 ? COLORS.sky : COLORS.red) }] }, options: baseOptions('Billing period', '% vs previous period', { scales: { x: { title: { display: true, text: 'Billing period' }, grid: { display: false } }, y: { title: { display: true, text: '% change' } } } }) }; });
  season = computed(() => { const a = this.a(); const cols = [COLORS.navy, COLORS.blue, COLORS.sky, COLORS.teal, COLORS.amber, COLORS.red];
    return lineCfg(a.seasonal.seasons, Object.entries(a.seasonal.series).map(([y, v], i) => ({ label: y, data: v, borderColor: cols[i % cols.length], tension: .2, spanGaps: false })), 'Billing season', a.unit); });
  ngOnInit() { this.load(); }
  load() { this.error.set(''); this.api.get('/dashboard/analytics', { dataset_id: this.datasetId(), start: this.start, end: this.end }).subscribe({ next: (r) => this.a.set(r), error: (e) => this.error.set(errText(e)) }); }
}

// ---------------------------------------------------------------- Predict
@Component({
  selector: 'd-predict',
  standalone: true,
  imports: [DatasetPick, DChart, FormsModule],
  template: `
  <div class="d-pagehead"><h2>Make Prediction</h2></div>
  <section class="d-panel">
    <div class="d-filters">
      <d-dataset-pick [(value)]="datasetId" [readyOnly]="true" />
      <label class="d-field">Model
        <select disabled><option>CNN-LSTM + Attention (deployed, saved weights)</option></select></label>
      <label class="d-field">Forecast horizon
        <select [(ngModel)]="horizon">@for (h of [1,2,3,4,5,6]; track h) { <option [ngValue]="h">{{ h }} billing period{{ h > 1 ? 's' : '' }}</option> }</select></label>
      <button class="d-btn primary" (click)="run()" [disabled]="busy()">{{ busy() ? 'Predicting…' : 'Predict' }}</button>
    </div>
    <p class="d-muted">Only the deployed model has saved weights, so it is the only one that can predict new data. The other models appear on the Model Comparison page with their saved evaluation results. The model predicts one billing period ahead; longer horizons feed its own forecasts back in (autoregressive) and are less reliable. Only datasets you confirmed on the Upload page appear here.</p>
  </section>
  @if (error()) { <div class="d-alert err" role="alert">{{ error() }}</div> }
  @if (r(); as r) {
    <section class="d-panel">
      <h3>Genuine future forecast <small class="d-muted">— no actual values exist yet</small></h3>
      <p class="d-muted">Model: <b>{{ r.model.label }}</b> · run {{ r.model.run_id }} · evaluated {{ when(r.model.evaluated_at) }} · predicted {{ when(r.created_at) }} · unit {{ r.unit }} ({{ r.unit_note }})</p>
      <div class="d-scroll"><table class="d-table"><thead><tr><th>Step</th><th>Billing period</th><th>Forecast</th><th>Range</th><th>Method</th></tr></thead><tbody>
        @for (x of r.forecast; track x.step) { <tr><td>{{ x.step }}</td><td>{{ x.label }}</td><td><b>{{ fmt(x.forecast) }}</b></td><td>{{ x.lower !== null ? fmt(x.lower) + ' – ' + fmt(x.upper) : 'not measured' }}</td><td>{{ x.method }}</td></tr> }</tbody></table></div>
      <p class="d-muted">{{ r.forecast_note }}</p>
      <d-chart [config]="fcChart()" [height]="300" label="History and forecast" />
      <div class="d-row"><button class="d-btn" (click)="csv()">Export CSV</button></div>
    </section>
    <section class="d-panel">
      <h3>Historical one-step predictions vs actual</h3>
      <div class="d-alert" [class.warn]="r.backtest.in_sample">{{ r.backtest.scope }}</div>
      @if (r.backtest.rows.length) {
        <div class="d-stats"><div>RMSE <b>{{ fmt(r.backtest.metrics.rmse) }}</b></div><div>MAE <b>{{ fmt(r.backtest.metrics.mae) }}</b></div><div>MSE <b>{{ fmt(r.backtest.metrics.mse) }}</b></div><div>R² <b>{{ fmt(r.backtest.metrics.r2, 3) }}</b></div><div>periods <b>{{ r.backtest.metrics.n }}</b></div></div>
        <d-chart [config]="btChart()" label="Actual versus predicted" />
        <div class="d-scroll"><table class="d-table"><thead><tr><th>Period</th><th>Actual</th><th>Predicted</th><th>Residual (actual − predicted)</th></tr></thead><tbody>
          @for (x of r.backtest.rows; track x.label) { <tr><td>{{ x.label }}</td><td>{{ fmt(x.actual) }}</td><td>{{ fmt(x.predicted) }}</td><td>{{ fmt(x.residual) }}</td></tr> }</tbody></table></div>
      } @else { <div class="d-empty">Not enough history for a backtest.</div> }
    </section>
  }`,
})
export class Predict {
  private api = inject(Dash); private toast = inject(Toasts);
  datasetId = signal(0); horizon = 1; busy = signal(false); error = signal(''); r = signal<any>(null);
  fmt = fmt; when = when;
  fcChart = computed(() => { const r = this.r(); const n = r.history.labels.length; const labels = [...r.history.labels, ...r.forecast.map((x: any) => x.label)];
    const hist = [...r.history.values, ...r.forecast.map(() => null)]; const fc = labels.map(() => null as number | null);
    fc[n - 1] = r.history.values[n - 1]; r.forecast.forEach((x: any, i: number) => (fc[n + i] = x.forecast));
    const lo = labels.map(() => null as number | null), hi = labels.map(() => null as number | null);
    r.forecast.forEach((x: any, i: number) => { lo[n + i] = x.lower; hi[n + i] = x.upper; });
    return lineCfg(labels, [{ label: 'Actual', data: hist, borderColor: COLORS.blue }, { label: 'Forecast', data: fc, borderColor: COLORS.amber, borderDash: [6, 4], pointRadius: 4 },
      { label: 'Lower (±typical error)', data: lo, borderColor: COLORS.grey, pointRadius: 2, borderDash: [2, 3] }, { label: 'Upper (±typical error)', data: hi, borderColor: COLORS.grey, pointRadius: 2, borderDash: [2, 3] }], 'Billing period', r.unit); });
  btChart = computed(() => { const r = this.r(); const rows = r.backtest.rows; return lineCfg(rows.map((x: any) => x.label), [
    { label: 'Actual', data: rows.map((x: any) => x.actual), borderColor: COLORS.blue }, { label: 'Predicted (one step)', data: rows.map((x: any) => x.predicted), borderColor: COLORS.amber }], 'Billing period', r.unit); });
  run() {
    this.busy.set(true); this.error.set('');
    this.api.post('/predictions', { dataset_id: this.datasetId(), horizon: this.horizon }).subscribe({
      next: (r) => { this.r.set(r); this.busy.set(false); this.toast.show('Prediction complete and saved to history.'); },
      error: (e) => { this.busy.set(false); this.error.set(errText(e)); } });
  }
  csv() { const r = this.r(); this.api.download(`/predictions/${r.prediction_id}/csv`, `prediction_${r.prediction_id}.csv`, (m) => this.toast.show(m, 'err')); }
}

// ---------------------------------------------------------------- History
@Component({
  selector: 'd-history',
  standalone: true,
  imports: [FormsModule],
  template: `
  <div class="d-pagehead"><h2>Prediction History</h2></div>
  <div class="d-filters"><label class="d-field">From<input type="date" [(ngModel)]="from" (change)="load()"></label>
    <label class="d-field">To<input type="date" [(ngModel)]="to" (change)="load()"></label>
    <label class="d-field">Model<select [(ngModel)]="model" (change)="load()"><option value="">All</option><option value="cnn_lstm_attention">cnn_lstm_attention</option></select></label></div>
  @if (error()) { <div class="d-alert err">{{ error() }}</div> }
  @if (rows().length === 0) { <div class="d-empty">No predictions yet. Run one on the Make Prediction page.</div> } @else {
  <div class="d-scroll"><table class="d-table"><thead><tr><th>#</th><th>When</th><th>Dataset</th><th>Model</th><th>Horizon</th><th>First forecast</th><th></th></tr></thead><tbody>
    @for (p of rows(); track p.id) { <tr><td>{{ p.id }}</td><td>{{ when(p.created_at) }}</td><td>{{ p.dataset_name }}</td><td>{{ p.model_variant }}</td><td>{{ p.horizon }}</td>
      <td>{{ p.first_forecast ? p.first_forecast.label + ': ' + fmt(p.first_forecast.forecast) : '—' }}</td>
      <td class="d-actions"><button class="d-btn sm" (click)="open(p.id)">Details</button><button class="d-btn sm" (click)="csv(p.id)">CSV</button></td></tr> }</tbody></table></div> }
  @if (detail(); as d) { <section class="d-panel"><h3>Prediction #{{ d.id }} — {{ d.dataset_name }}</h3>
    <p class="d-muted">{{ d.result.model.variant }} · {{ when(d.created_at) }}</p>
    <table class="d-table"><thead><tr><th>Period</th><th>Forecast</th><th>Range</th><th>Method</th></tr></thead><tbody>
      @for (x of d.result.forecast; track x.step) { <tr><td>{{ x.label }}</td><td>{{ fmt(x.forecast) }}</td><td>{{ x.lower !== null ? fmt(x.lower) + ' – ' + fmt(x.upper) : 'not measured' }}</td><td>{{ x.method }}</td></tr> }</tbody></table>
    <p class="d-muted">{{ d.result.backtest.scope }}</p>
    <p class="d-muted">Future periods have no actual consumption yet, so predictions cannot be compared with actuals until that data is uploaded and a new prediction is run.</p></section> }`,
})
export class History implements OnInit {
  private api = inject(Dash); private toast = inject(Toasts);
  rows = signal<any[]>([]); detail = signal<any>(null); error = signal(''); from = ''; to = ''; model = ''; fmt = fmt; when = when;
  ngOnInit() { this.load(); }
  load() { this.api.get('/predictions', { date_from: this.from, date_to: this.to, model: this.model }).subscribe({ next: (r) => this.rows.set(r.predictions), error: (e) => this.error.set(errText(e)) }); }
  open(id: number) { this.api.get(`/predictions/${id}`).subscribe({ next: (r) => this.detail.set(r), error: (e) => this.toast.show(errText(e), 'err') }); }
  csv(id: number) { this.api.download(`/predictions/${id}/csv`, `prediction_${id}.csv`, (m) => this.toast.show(m, 'err')); }
}

// ---------------------------------------------------------------- Reports
@Component({
  selector: 'd-reports',
  standalone: true,
  template: `
  <div class="d-pagehead"><h2>Reports and Downloads</h2></div>
  <p class="d-muted">Every report is generated from the saved results of the trained model run — nothing is estimated.</p>
  <div class="d-grid2">
    @for (r of reports; track r.path) { <section class="d-panel"><h3>{{ r.title }}</h3><p class="d-muted">{{ r.desc }}</p><button class="d-btn primary" (click)="get(r)">Download CSV</button></section> }
  </div>`,
})
export class Reports {
  private api = inject(Dash); private toast = inject(Toasts);
  reports = [
    { title: 'Model metrics', desc: 'RMSE, MAE, MSE, R² and MAPE for all four neural variants and both baselines, with weight availability.', path: '/reports/model-metrics.csv', file: 'model_metrics.csv' },
    { title: 'Walk-forward predictions', desc: 'Actual and predicted values of every model on each held-out test bill.', path: '/reports/walk-forward.csv', file: 'walk_forward_predictions.csv' },
    { title: 'Model validation summary', desc: 'Every validation check with its status and reason.', path: '/reports/validation.csv', file: 'model_validation.csv' },
    { title: 'Prediction results', desc: 'Download an individual prediction from the Prediction History page.', path: '', file: '' },
  ];
  get(r: { path: string; file: string }) { if (!r.path) { this.toast.show('Open Prediction History and press CSV on a row.'); return; } this.api.download(r.path, r.file, (m) => this.toast.show(m, 'err')); }
}

// ---------------------------------------------------------------- Profile
@Component({
  selector: 'd-profile',
  standalone: true,
  imports: [FormsModule],
  template: `
  <div class="d-pagehead"><h2>Profile and Settings</h2></div>
  @if (auth.user(); as u) { <section class="d-panel"><h3>Account</h3>
    <dl class="d-dl"><dt>Username</dt><dd>{{ u.username }}</dd><dt>Email</dt><dd>{{ u.email || '—' }}</dd><dt>Role</dt><dd>{{ u.role }}</dd><dt>Created</dt><dd>{{ when(u.created_at) }}</dd><dt>Last sign-in</dt><dd>{{ when(u.last_login) }}</dd></dl></section> }
  <section class="d-panel"><h3>Change password</h3>
    <form (ngSubmit)="change()" class="d-form"><label class="d-field">Current password<input type="password" name="c" [(ngModel)]="current" autocomplete="current-password"></label>
      <label class="d-field">New password (10+ characters, letters and digits)<input type="password" name="n" [(ngModel)]="next" autocomplete="new-password"></label>
      <button class="d-btn primary" type="submit">Update password</button></form></section>
  <section class="d-panel"><h3>Metric guide</h3>
    <dl class="d-dl"><dt>RMSE</dt><dd>Root mean squared error, in the data's unit. Lower is better; large misses count more.</dd>
      <dt>MAE</dt><dd>Average absolute miss, in the data's unit. Lower is better.</dd>
      <dt>MSE</dt><dd>Mean squared error (RMSE²). Lower is better; the unit is squared.</dd>
      <dt>R²</dt><dd>Share of variation explained relative to predicting the test-period average. 1 is perfect; below 0 means worse than that simple guess.</dd>
      <dt>MAPE</dt><dd>Average error as a percentage of the actual value (valid because consumption is always positive).</dd>
      <dt>Walk-forward</dt><dd>Each test bill is predicted by a model that only saw earlier bills — the honest accuracy test used here.</dd>
      <dt>In-sample</dt><dd>Errors on data the model was trained on; optimistic, not a measure of forecasting skill.</dd></dl></section>`,
})
export class Profile {
  auth = inject(Auth); private api = inject(Dash); private toast = inject(Toasts);
  current = ''; next = ''; when = when;
  change() { this.api.post('/auth/change-password', { current_password: this.current, new_password: this.next }).subscribe({ next: () => { this.toast.show('Password updated.'); this.current = this.next = ''; }, error: (e) => this.toast.show(errText(e), 'err') }); }
}
