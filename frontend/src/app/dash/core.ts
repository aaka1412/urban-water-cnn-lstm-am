import { Component, ElementRef, Injectable, OnDestroy, effect, inject, input, signal, viewChild } from '@angular/core';
import { HttpClient, HttpErrorResponse, HttpEvent, HttpEventType, HttpInterceptorFn } from '@angular/common/http';
import { Router, CanActivateFn } from '@angular/router';
import { Chart, ChartConfiguration, registerables } from 'chart.js';
import { Observable, tap } from 'rxjs';

Chart.register(...registerables);

export const BASE = typeof window !== 'undefined' && window.location.port === '4200' ? 'http://localhost:8000/api' : '/api';
const TOKEN_KEY = 'water_dash_token';

/** Human-readable message from any backend error. */
export function errText(e: unknown): string {
  if (e instanceof HttpErrorResponse) {
    if (e.status === 0) return 'Cannot reach the backend. Is it running on port 8000?';
    const d = e.error?.detail;
    if (typeof d === 'string') return d;
    if (Array.isArray(d)) return d.map((x: any) => x.msg).join('; ');
    return `Request failed (${e.status}).`;
  }
  return 'Unexpected error.';
}

export const fmt = (n: number | null | undefined, digits = 0) =>
  n === null || n === undefined ? '—' : new Intl.NumberFormat('en-IN', { maximumFractionDigits: digits, minimumFractionDigits: digits }).format(n);
export const compact = (v: number | string) => {
  const n = Number(v);
  if (Math.abs(n) >= 1e7) return (n / 1e7).toFixed(1) + ' Cr';
  if (Math.abs(n) >= 1e5) return (n / 1e5).toFixed(1) + ' L';
  if (Math.abs(n) >= 1e3) return (n / 1e3).toFixed(0) + 'k';
  return String(n);
};
export const when = (iso: string | null | undefined) => (iso ? new Date(iso).toLocaleString() : '—');

// ----------------------------------------------------------------- toasts
@Injectable({ providedIn: 'root' })
export class Toasts {
  readonly items = signal<{ id: number; kind: 'ok' | 'err'; text: string }[]>([]);
  private n = 0;
  show(text: string, kind: 'ok' | 'err' = 'ok') {
    const id = ++this.n;
    this.items.update((l) => [...l, { id, kind, text }]);
    setTimeout(() => this.items.update((l) => l.filter((t) => t.id !== id)), kind === 'err' ? 7000 : 3500);
  }
}

// ----------------------------------------------------------------- auth
export interface User { id: number; username: string; email: string | null; role: 'user' | 'admin'; created_at: string; last_login: string | null; }

@Injectable({ providedIn: 'root' })
export class Auth {
  private http = inject(HttpClient);
  private router = inject(Router);
  readonly user = signal<User | null>(null);
  token(): string | null { return typeof sessionStorage === 'undefined' ? null : sessionStorage.getItem(TOKEN_KEY); }

  private store(r: { token: string; user: User }) { sessionStorage.setItem(TOKEN_KEY, r.token); this.user.set(r.user); }
  login(username: string, password: string, admin = false) {
    return this.http.post<{ token: string; user: User }>(`${BASE}/auth/${admin ? 'admin/' : ''}login`, { username, password }).pipe(tap((r) => this.store(r)));
  }
  register(username: string, password: string, email: string | null) {
    return this.http.post(`${BASE}/auth/register`, { username, password, email: email || null });
  }
  /** Restore the session after a page reload; resolves to the user or null. */
  restore(): Promise<User | null> {
    if (this.user()) return Promise.resolve(this.user());
    if (!this.token()) return Promise.resolve(null);
    return new Promise((resolve) =>
      this.http.get<{ user: User }>(`${BASE}/auth/me`).subscribe({
        next: (r) => { this.user.set(r.user); resolve(r.user); },
        error: () => { this.clear(); resolve(null); },
      }));
  }
  clear() { sessionStorage.removeItem(TOKEN_KEY); this.user.set(null); }
  logout() {
    const done = () => { this.clear(); this.router.navigateByUrl('/login'); };
    this.http.post(`${BASE}/auth/logout`, {}).subscribe({ next: done, error: done });
  }
}

export const authInterceptor: HttpInterceptorFn = (req, next) => {
  const auth = inject(Auth);
  const router = inject(Router);
  const t = auth.token();
  const authed = t && req.url.startsWith(BASE) ? req.clone({ setHeaders: { Authorization: `Bearer ${t}` } }) : req;
  return next(authed).pipe(
    tap({
      error: (e) => {
        if (e instanceof HttpErrorResponse && e.status === 401 && !req.url.includes('/auth/') && auth.token()) {
          auth.clear();
          router.navigateByUrl('/login');
        }
      },
    }),
  );
};

export const authGuard: CanActivateFn = async () => {
  const auth = inject(Auth), router = inject(Router);
  return (await auth.restore()) ? true : router.parseUrl('/login');
};
export const adminGuard: CanActivateFn = async () => {
  const auth = inject(Auth), router = inject(Router);
  const u = await auth.restore();
  return u?.role === 'admin' ? true : router.parseUrl('/admin/login');
};

// ----------------------------------------------------------------- API
@Injectable({ providedIn: 'root' })
export class Dash {
  private http = inject(HttpClient);
  get = <T = any>(path: string, params: Record<string, string | number | null | undefined> = {}) => {
    const p: Record<string, string> = {};
    for (const [k, v] of Object.entries(params)) if (v !== null && v !== undefined && v !== '') p[k] = String(v);
    return this.http.get<T>(`${BASE}${path}`, { params: p });
  };
  post = <T = any>(path: string, body: unknown = {}) => this.http.post<T>(`${BASE}${path}`, body);
  patch = <T = any>(path: string, body: unknown) => this.http.patch<T>(`${BASE}${path}`, body);
  del = <T = any>(path: string) => this.http.delete<T>(`${BASE}${path}`);
  upload(file: File): Observable<HttpEvent<any>> {
    const fd = new FormData();
    fd.append('file', file, file.name);
    return this.http.post(`${BASE}/datasets/upload`, fd, { reportProgress: true, observe: 'events' });
  }
  /** Authenticated file download (fetches as blob so the Authorization header is sent). */
  download(path: string, filename: string, onError: (m: string) => void) {
    this.http.get(`${BASE}${path}`, { responseType: 'blob' }).subscribe({
      next: (blob) => {
        const a = document.createElement('a');
        a.href = URL.createObjectURL(blob);
        a.download = filename;
        a.click();
        URL.revokeObjectURL(a.href);
      },
      error: (e) => onError(errText(e)),
    });
  }
}
export { HttpEventType };

// ----------------------------------------------------------------- chart
export const COLORS = { blue: '#1565c0', sky: '#42a5f5', navy: '#0d3b66', teal: '#00897b', amber: '#f59e0b', red: '#d64545', grey: '#9aa7b5', green: '#2e9d62' };

export function baseOptions(xTitle: string, yTitle: string, extra: any = {}): any {
  return {
    responsive: true, maintainAspectRatio: false, animation: false, interaction: { mode: 'index', intersect: false },
    plugins: { legend: { position: 'bottom', labels: { boxWidth: 12, usePointStyle: true } }, tooltip: { callbacks: { label: (c: any) => ` ${c.dataset.label}: ${fmt(c.parsed.y ?? c.parsed, 2)}` } } },
    scales: {
      x: { title: { display: !!xTitle, text: xTitle }, grid: { display: false }, ticks: { maxRotation: 60, autoSkip: true } },
      y: { title: { display: !!yTitle, text: yTitle }, ticks: { callback: (v: any) => compact(v) } },
    },
    ...extra,
  };
}

@Component({
  selector: 'd-chart',
  standalone: true,
  template: `<div class="d-chartbox" [style.height.px]="height()"><canvas #c role="img" [attr.aria-label]="label()"></canvas></div>`,
})
export class DChart implements OnDestroy {
  config = input.required<ChartConfiguration>();
  height = input(300);
  label = input('Chart');
  private canvas = viewChild.required<ElementRef<HTMLCanvasElement>>('c');
  private inst: Chart | null = null;
  constructor() {
    effect(() => {
      const cfg = this.config();
      const el = this.canvas().nativeElement;
      this.inst?.destroy();
      this.inst = new Chart(el, cfg);
    });
  }
  ngOnDestroy() { this.inst?.destroy(); }
}
