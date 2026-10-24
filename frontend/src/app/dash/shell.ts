import { Component, inject, signal } from '@angular/core';
import { Router, RouterLink, RouterLinkActive, RouterOutlet, NavigationEnd } from '@angular/router';
import { FormsModule } from '@angular/forms';
import { toSignal } from '@angular/core/rxjs-interop';
import { filter, map } from 'rxjs';
import { Auth, Toasts, errText } from './core';

/** Bootstrapped root: router outlet + toasts + a way back from the (unchanged) chatbot page. */
@Component({
  selector: 'app-root',
  standalone: true,
  imports: [RouterOutlet, RouterLink],
  template: `
    <router-outlet />
    @if (onChat()) { <a class="d-backpill" routerLink="/overview">← Dashboard</a> }
    <div class="d-toasts" aria-live="polite">
      @for (t of toasts.items(); track t.id) { <div class="d-toast" [class.err]="t.kind === 'err'">{{ t.text }}</div> }
    </div>`,
})
export class Root {
  toasts = inject(Toasts);
  private router = inject(Router);
  onChat = toSignal(this.router.events.pipe(filter((e) => e instanceof NavigationEnd), map(() => this.router.url.startsWith('/chatbot'))), { initialValue: false });
}

@Component({
  selector: 'd-shell',
  standalone: true,
  imports: [RouterOutlet, RouterLink, RouterLinkActive],
  template: `
  <div class="dash d-layout" [class.open]="open()">
    <aside class="d-side" aria-label="Main navigation">
      <div class="d-brand"><span class="d-logo">💧</span><div><b>Urban Water</b><small>Consumption Analytics</small></div></div>
      <nav>
        @for (n of nav; track n.path) {
          <a [routerLink]="n.path" routerLinkActive="on" (click)="open.set(false)"><span aria-hidden="true">{{ n.icon }}</span>{{ n.label }}</a>
        }
        @if (auth.user()?.role === 'admin') {
          <div class="d-navsep">Administration</div>
          <a routerLink="/admin" routerLinkActive="on" (click)="open.set(false)"><span aria-hidden="true">🛡️</span>Admin Dashboard</a>
        }
        <button class="d-navbtn" (click)="auth.logout()"><span aria-hidden="true">⎋</span>Logout</button>
      </nav>
    </aside>
    <div class="d-main">
      <header class="d-top">
        <button class="d-burger" (click)="open.set(!open())" aria-label="Toggle navigation">☰</button>
        <h1>Urban Water Consumption Prediction and Analytics System</h1>
        <span class="d-grow"></span>
        <a class="d-btn primary" routerLink="/upload">⬆ Upload Dataset</a>
        <a class="d-user" routerLink="/profile" title="Profile">{{ auth.user()?.username }}</a>
      </header>
      <main class="d-content"><router-outlet /></main>
    </div>
    @if (open()) { <div class="d-scrim" (click)="open.set(false)"></div> }
  </div>`,
})
export class Shell {
  auth = inject(Auth);
  open = signal(false);
  nav = [
    { path: '/overview', label: 'Overview', icon: '▦' },
    { path: '/upload', label: 'Upload Dataset', icon: '⬆' },
    { path: '/analytics', label: 'Water Consumption Analytics', icon: '📈' },
    { path: '/predict', label: 'Make Prediction', icon: '🔮' },
    { path: '/models', label: 'Model Comparison', icon: '⚖' },
    { path: '/training', label: 'Training and Validation', icon: '🧪' },
    { path: '/confusion', label: 'Confusion Matrix', icon: '▤' },
    { path: '/history', label: 'Prediction History', icon: '🕘' },
    { path: '/chatbot', label: 'Chatbot', icon: '💬' },
    { path: '/reports', label: 'Reports and Downloads', icon: '⬇' },
    { path: '/profile', label: 'Profile and Settings', icon: '⚙' },
  ];
}

@Component({
  selector: 'd-login',
  standalone: true,
  imports: [FormsModule, RouterLink],
  template: `
  <div class="dash d-authpage">
    <form class="d-authcard" (ngSubmit)="submit()" autocomplete="on">
      <div class="d-logo big">💧</div>
      <h2>{{ admin ? 'Administrator sign in' : (mode() === 'login' ? 'Sign in' : 'Create an account') }}</h2>
      <p class="d-muted">Urban Water Consumption Prediction and Analytics System</p>
      <label>Username<input name="u" [(ngModel)]="username" required minlength="3" autocomplete="username"></label>
      @if (mode() === 'register') { <label>Email (optional)<input name="e" type="email" [(ngModel)]="email" autocomplete="email"></label> }
      <label>Password<input name="p" type="password" [(ngModel)]="password" required autocomplete="current-password"></label>
      @if (mode() === 'register') { <small class="d-muted">At least 10 characters, with letters and digits.</small> }
      @if (error()) { <div class="d-alert err" role="alert">{{ error() }}</div> }
      @if (info()) { <div class="d-alert ok">{{ info() }}</div> }
      <button class="d-btn primary wide" type="submit" [disabled]="busy()">{{ busy() ? 'Please wait…' : (mode() === 'login' ? 'Sign in' : 'Register') }}</button>
      @if (!admin) {
        <button type="button" class="d-link" (click)="toggle()">{{ mode() === 'login' ? 'New here? Create an account' : 'Have an account? Sign in' }}</button>
        <a class="d-link" routerLink="/admin/login">Administrator sign in</a>
      } @else { <a class="d-link" routerLink="/login">← User sign in</a> }
    </form>
  </div>`,
})
export class Login {
  private auth = inject(Auth);
  private router = inject(Router);
  admin = this.router.url.startsWith('/admin');
  mode = signal<'login' | 'register'>('login');
  username = ''; password = ''; email = '';
  error = signal(''); info = signal(''); busy = signal(false);
  toggle() { this.mode.update((m) => (m === 'login' ? 'register' : 'login')); this.error.set(''); this.info.set(''); }
  submit() {
    this.error.set(''); this.info.set('');
    if (!this.username || !this.password) { this.error.set('Enter a username and password.'); return; }
    this.busy.set(true);
    const done = () => this.busy.set(false);
    if (this.mode() === 'register') {
      this.auth.register(this.username, this.password, this.email).subscribe({
        next: () => { done(); this.info.set('Account created. You can sign in now.'); this.mode.set('login'); this.password = ''; },
        error: (e) => { done(); this.error.set(errText(e)); },
      });
      return;
    }
    this.auth.login(this.username, this.password, this.admin).subscribe({
      next: () => { done(); this.router.navigateByUrl(this.admin ? '/admin' : '/overview'); },
      error: (e) => { done(); this.error.set(errText(e)); },
    });
  }
}
