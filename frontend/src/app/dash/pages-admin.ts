import { Component, OnInit, inject, signal } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { Auth, Dash, Toasts, errText, fmt, when } from './core';
import { StatusChip } from './pages-user';

@Component({
  selector: 'd-admin',
  standalone: true,
  imports: [FormsModule, StatusChip],
  template: `
  <div class="d-pagehead"><h2>Admin Dashboard</h2></div>
  <div class="d-tabs" role="tablist">@for (t of tabs; track t[0]) { <button role="tab" [class.on]="tab() === t[0]" (click)="go(t[0])">{{ t[1] }}</button> }</div>
  @if (error()) { <div class="d-alert err">{{ error() }}</div> }

  @if (tab() === 'overview' && ov(); as o) {
    <section class="d-cards">
      <div class="d-card"><span>Datasets uploaded</span><b>{{ o.counts.datasets_uploaded }}</b><small>{{ o.counts.datasets_rejected }} rejected</small></div>
      <div class="d-card"><span>Users</span><b>{{ o.counts.users }}</b></div>
      <div class="d-card"><span>Prediction requests</span><b>{{ o.counts.predictions }}</b></div>
      <div class="d-card"><span>Models with saved weights</span><b>{{ o.counts.trained_models_with_weights }}</b><small>{{ o.counts.models_with_evaluation }} models have evaluation results</small></div>
      <div class="d-card"><span>Latest validation status</span><b><d-status [status]="o.latest_evaluation.validation_overall" /></b><small>Best by RMSE: {{ o.latest_evaluation.best_model_by_rmse?.label }}</small></div>
      <div class="d-card"><span>Last re-evaluation</span><b class="sm">{{ when(o.latest_evaluation.last_recomputed) }}</b></div>
    </section>
    <div class="d-grid2">
      <section class="d-panel"><h3>Recent dataset uploads</h3>@if (!o.recent_datasets.length) { <div class="d-empty">None yet.</div> } @else { <table class="d-table"><tbody>@for (d of o.recent_datasets; track d.id) { <tr><td>{{ d.name }}</td><td><d-status [status]="d.status" /></td><td>{{ when(d.uploaded_at) }}</td></tr> }</tbody></table> }</section>
      <section class="d-panel"><h3>Recent prediction activity</h3>@if (!o.recent_predictions.length) { <div class="d-empty">None yet.</div> } @else { <table class="d-table"><tbody>@for (p of o.recent_predictions; track p.id) { <tr><td>#{{ p.id }} {{ p.dataset_name }}</td><td>h={{ p.horizon }}</td><td>{{ when(p.created_at) }}</td></tr> }</tbody></table> }</section>
    </div>
    <section class="d-panel"><h3>Recent errors and validation failures</h3>@if (!o.recent_errors.length) { <div class="d-empty">No failures recorded.</div> } @else {
      <div class="d-scroll"><table class="d-table"><thead><tr><th>When</th><th>User</th><th>Action</th><th>Outcome</th><th>Detail</th></tr></thead><tbody>@for (e of o.recent_errors; track $index) { <tr><td>{{ when(e.ts) }}</td><td>{{ e.username }}</td><td>{{ e.action }}</td><td><d-status [status]="e.outcome" /></td><td>{{ e.detail }}</td></tr> }</tbody></table></div> }</section>
  }

  @if (tab() === 'datasets' && ds(); as d) {
    <section class="d-panel"><h3>Datasets</h3>
      <table class="d-table"><thead><tr><th>Name</th><th>Owner</th><th>Uploaded</th><th>Rows</th><th>Validation</th><th>Status</th><th></th></tr></thead><tbody>
        <tr><td><b>{{ d.original.name }}</b></td><td>—</td><td>—</td><td>{{ d.original.n_rows }}</td><td><d-status status="passed" /></td><td><span class="d-chip s-ready">Protected</span></td><td></td></tr>
        @for (x of d.datasets; track x.id) { <tr><td>{{ x.name }}</td><td>{{ x.owner }}</td><td>{{ when(x.uploaded_at) }}</td><td>{{ x.n_rows ?? '—' }}</td><td><d-status [status]="x.validation_status" /></td><td><d-status [status]="x.status" /></td>
          <td class="d-actions">@if (x.status !== 'rejected' && x.status !== 'deleted') { <button class="d-btn sm" (click)="preview(x.id)">Preview</button> }
            <button class="d-btn sm" (click)="errorsOf(x.id)">Validation</button>
            @if (x.status !== 'deleted') { @if (delId() === x.id) { <button class="d-btn sm danger" (click)="remove(x.id)">Confirm delete</button> <button class="d-btn sm" (click)="delId.set(0)">Cancel</button> } @else { <button class="d-btn sm danger" (click)="delId.set(x.id)">Delete</button> } }</td></tr> }</tbody></table></section>
    @if (pv(); as p) { <section class="d-panel"><h3>Preview: {{ p.dataset.name }}</h3><div class="d-scroll"><table class="d-table"><thead><tr>@for (k of p.columns; track k) { <th>{{ k }}</th> }</tr></thead><tbody>@for (r of p.rows; track $index) { <tr>@for (k of p.columns; track k) { <td>{{ r[k] }}</td> }</tr> }</tbody></table></div></section> }
    @if (val(); as v) { <section class="d-panel"><h3>Validation details <d-status [status]="v.validation.status" /></h3>
      @for (e of v.validation.errors; track e) { <div class="d-alert err">{{ e }}</div> } @for (w of v.validation.warnings; track w) { <div class="d-alert warn">{{ w }}</div> }
      <table class="d-table"><tbody>@for (c of v.validation.checks; track c.name) { <tr><td>{{ c.name }}</td><td><d-status [status]="c.status" /></td><td>{{ c.detail }}</td></tr> }</tbody></table></section> }
  }

  @if (tab() === 'models' && mi(); as m) {
    <section class="d-panel"><h3>Model information</h3>
      @if (m.loaded) { <dl class="d-dl"><dt>Loading status</dt><dd><span class="d-chip s-passed">Loaded</span></dd><dt>Deployed variant</dt><dd>{{ m.variant }}</dd><dt>Run (version)</dt><dd>{{ m.run_id }}</dd><dt>Weights file</dt><dd>{{ m.weights_file }} ({{ fmt(m.weights_bytes) }} bytes)</dd><dt>Window</dt><dd>{{ m.sequence_length }} billing periods</dd><dt>Target</dt><dd>{{ m.target }} ({{ m.units }})</dd><dt>Evaluated</dt><dd>{{ when(m.evaluated_at) }}</dd><dt>Available runs</dt><dd>{{ m.available_runs.join(', ') }}</dd></dl> }
      @else { <div class="d-alert err">Model not loaded: {{ m.error }}</div> }
      <div class="d-alert">{{ m.retraining.reason }}</div>
      <button class="d-btn primary" (click)="evaluate()" [disabled]="busy()">Re-evaluate (recompute metrics from saved predictions)</button></section>
    <section class="d-panel"><h3>Latest evaluation</h3><div class="d-scroll"><table class="d-table"><thead><tr><th>Model</th><th>Weights</th><th>RMSE</th><th>MAE</th><th>R²</th></tr></thead><tbody>
      @for (c of m.catalog; track c.key) { <tr><td>{{ c.label }}</td><td>{{ c.has_saved_weights ? 'saved' : 'no' }}</td><td>{{ fmt(c.rmse) }}</td><td>{{ fmt(c.mae) }}</td><td>{{ fmt(c.r2, 3) }}</td></tr> }</tbody></table></div>
      @if (m.evaluations.length) { <p class="d-muted">Re-evaluations run: {{ m.evaluations.length }} (latest {{ when(m.evaluations[0].ts) }}).</p> }</section>
  }

  @if (tab() === 'predictions' && pr(); as p) {
    <div class="d-filters"><label class="d-field">From<input type="date" [(ngModel)]="from" (change)="loadPreds()"></label><label class="d-field">To<input type="date" [(ngModel)]="to" (change)="loadPreds()"></label>
      <label class="d-field">Model<select [(ngModel)]="model" (change)="loadPreds()"><option value="">All</option><option value="cnn_lstm_attention">cnn_lstm_attention</option></select></label>
      <button class="d-btn" (click)="exportPreds()">Export CSV</button></div>
    <section class="d-panel">@if (!p.length) { <div class="d-empty">No predictions match.</div> } @else { <div class="d-scroll"><table class="d-table"><thead><tr><th>#</th><th>User</th><th>When</th><th>Dataset</th><th>Model</th><th>Horizon</th><th>First forecast</th></tr></thead><tbody>
      @for (x of p; track x.id) { <tr><td>{{ x.id }}</td><td>{{ x.username }}</td><td>{{ when(x.created_at) }}</td><td>{{ x.dataset_name }}</td><td>{{ x.model_variant }}</td><td>{{ x.horizon }}</td><td>{{ x.first_forecast ? x.first_forecast.label + ': ' + fmt(x.first_forecast.forecast) : '—' }}</td></tr> }</tbody></table></div> }</section>
  }

  @if (tab() === 'users' && us(); as u) {
    <section class="d-panel"><h3>Users</h3><div class="d-scroll"><table class="d-table"><thead><tr><th>Username</th><th>Email</th><th>Role</th><th>Active</th><th>Last sign-in</th><th></th></tr></thead><tbody>
      @for (x of u; track x.id) { <tr><td>{{ x.username }}</td><td>{{ x.email || '—' }}</td><td>{{ x.role }}</td><td>{{ x.active ? 'yes' : 'no' }}</td><td>{{ when(x.last_login) }}</td>
        <td class="d-actions">@if (x.id !== me.user()?.id) { <button class="d-btn sm" (click)="patch(x.id, { role: x.role === 'admin' ? 'user' : 'admin' })">Make {{ x.role === 'admin' ? 'user' : 'admin' }}</button> <button class="d-btn sm" (click)="patch(x.id, { active: !x.active })">{{ x.active ? 'Deactivate' : 'Activate' }}</button> } @else { <small class="d-muted">you</small> }</td></tr> }</tbody></table></div></section>
  }

  @if (tab() === 'audit' && au(); as a) {
    <div class="d-filters"><label class="d-field">Outcome<select [(ngModel)]="outcome" (change)="loadAudit()"><option value="">All</option><option value="ok">ok</option><option value="failed">failed</option><option value="rejected">rejected</option><option value="denied">denied</option></select></label>
      <button class="d-btn" (click)="auditCsv()">Export CSV</button></div>
    <section class="d-panel"><div class="d-scroll"><table class="d-table"><thead><tr><th>When</th><th>User</th><th>Action</th><th>Outcome</th><th>Detail</th><th>IP</th></tr></thead><tbody>
      @for (e of a; track e.id) { <tr><td>{{ when(e.ts) }}</td><td>{{ e.username || '—' }}</td><td>{{ e.action }}</td><td><d-status [status]="e.outcome" /></td><td>{{ e.detail }}</td><td>{{ e.ip }}</td></tr> }</tbody></table></div></section>
  }`,
})
export class Admin implements OnInit {
  private api = inject(Dash); private toast = inject(Toasts); me = inject(Auth);
  tabs = [['overview', 'Overview'], ['datasets', 'Datasets'], ['models', 'Models'], ['predictions', 'Predictions'], ['users', 'Users'], ['audit', 'Audit log']];
  tab = signal('overview'); error = signal(''); busy = signal(false); delId = signal(0);
  ov = signal<any>(null); ds = signal<any>(null); mi = signal<any>(null); pr = signal<any[] | null>(null); us = signal<any[] | null>(null); au = signal<any[] | null>(null);
  pv = signal<any>(null); val = signal<any>(null);
  from = ''; to = ''; model = ''; outcome = ''; fmt = fmt; when = when;
  ngOnInit() { this.go('overview'); }
  go(t: string) {
    this.tab.set(t); this.error.set('');
    const fail = (e: unknown) => this.error.set(errText(e));
    if (t === 'overview') this.api.get('/admin/overview').subscribe({ next: (r) => this.ov.set(r), error: fail });
    if (t === 'datasets') this.api.get('/admin/datasets').subscribe({ next: (r) => this.ds.set(r), error: fail });
    if (t === 'models') this.api.get('/admin/models').subscribe({ next: (r) => this.mi.set(r), error: fail });
    if (t === 'predictions') this.loadPreds();
    if (t === 'users') this.api.get('/admin/users').subscribe({ next: (r) => this.us.set(r.users), error: fail });
    if (t === 'audit') this.loadAudit();
  }
  loadPreds() { this.api.get('/admin/predictions', { date_from: this.from, date_to: this.to, model: this.model }).subscribe({ next: (r) => this.pr.set(r.predictions), error: (e) => this.error.set(errText(e)) }); }
  loadAudit() { this.api.get('/admin/audit', { outcome: this.outcome }).subscribe({ next: (r) => this.au.set(r.entries), error: (e) => this.error.set(errText(e)) }); }
  exportPreds() { this.api.download('/admin/predictions.csv', 'predictions_export.csv', (m) => this.toast.show(m, 'err')); }
  auditCsv() { this.api.download('/admin/audit.csv', 'audit_log.csv', (m) => this.toast.show(m, 'err')); }
  preview(id: number) { this.api.get(`/datasets/${id}/preview`).subscribe({ next: (r) => this.pv.set(r), error: (e) => this.toast.show(errText(e), 'err') }); }
  errorsOf(id: number) { this.api.get(`/datasets/${id}/validation`).subscribe({ next: (r) => this.val.set(r), error: (e) => this.toast.show(errText(e), 'err') }); }
  remove(id: number) { this.api.del(`/datasets/${id}`).subscribe({ next: () => { this.delId.set(0); this.toast.show('Dataset deleted.'); this.go('datasets'); }, error: (e) => this.toast.show(errText(e), 'err') }); }
  evaluate() { this.busy.set(true); this.api.post('/admin/models/evaluate').subscribe({ next: () => { this.busy.set(false); this.toast.show('Metrics recomputed from saved predictions.'); this.go('models'); }, error: (e) => { this.busy.set(false); this.toast.show(errText(e), 'err'); } }); }
  patch(id: number, body: object) { this.api.patch(`/admin/users/${id}`, body).subscribe({ next: () => { this.toast.show('User updated.'); this.go('users'); }, error: (e) => this.toast.show(errText(e), 'err') }); }
}
