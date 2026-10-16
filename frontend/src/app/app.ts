import { Component, ElementRef, OnInit, ViewChild, computed, effect, inject, signal } from '@angular/core';
import { Subscription } from 'rxjs';
import { ApiService, ChatMessage, RunDetail, RunInfo } from './api.service';
import { ChartComponent } from './chart.component';
import { renderMarkdown } from './markdown';

interface Conversation {
  id: string;
  title: string;
  sessionId: string | null;
  runId: string | null;
  messages: ChatMessage[];
  updated: number;
}

type Lang = 'auto' | 'en' | 'hinglish';
type Theme = 'system' | 'light' | 'dark';

const LS_KEY = 'jalforecast.conversations.v2';
const LS_OLD_KEY = 'jalforecast.conversations.v1';
const LS_PREFS = 'jalforecast.prefs.v1';

const uid = () => Math.random().toString(36).slice(2, 10);

const FOLLOW_UPS: Record<string, { billing: string[]; daily: string[] }> = {
  forecast: { billing: ['Explain this forecast', 'Graph forecast 3 periods', 'How accurate is the model?'], daily: ['Explain this forecast', 'Graph forecast 7 days', 'How accurate is the model?'] },
  recent: { billing: ['Graph last 6 periods', 'Average last 6 periods', 'Forecast next billing cycle'], daily: ['Graph last 30 days', 'Explain this', 'Forecast 7 days'] },
  average: { billing: ['Highest period', 'Lowest period', 'Forecast next billing cycle'], daily: ['Peak day', 'Lowest day', 'Forecast 7 days'] },
  peak: { billing: ['Lowest period', 'Graph last 8 periods', 'Forecast next billing cycle'], daily: ['Lowest day', 'Graph last 30 days', 'Forecast 7 days'] },
  lowest: { billing: ['Highest period', 'Graph last 8 periods', 'Forecast next billing cycle'], daily: ['Peak day', 'Graph last 30 days', 'Forecast 7 days'] },
  metrics: { billing: ['Graph model comparison', 'How does the model work?', 'Forecast next billing cycle'], daily: ['Graph model comparison', 'How does the model work?', 'Forecast 7 days'] },
};

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [ChartComponent],
  templateUrl: './app.html',
  styleUrl: './app.scss',
})
export class App implements OnInit {
  private api = inject(ApiService);
  @ViewChild('thread') thread?: ElementRef<HTMLElement>;
  @ViewChild('box') box?: ElementRef<HTMLTextAreaElement>;

  // -------------------------------------------------------------- state
  readonly runs = signal<RunInfo[]>([]);
  readonly runId = signal<string | null>(null);
  readonly runDetail = signal<RunDetail | null>(null);
  readonly backendOk = signal<boolean | null>(null);
  readonly loadError = signal('');

  readonly conversations = signal<Conversation[]>([]);
  readonly activeId = signal('');
  readonly input = signal('');
  readonly lang = signal<Lang>('auto');
  readonly theme = signal<Theme>('system');
  readonly sidebarOpen = signal(typeof window === 'undefined' || window.innerWidth >= 900);
  readonly showInfo = signal(false);
  readonly copiedId = signal<string | null>(null);

  /** Conversation id waiting on the backend (typing indicator is per conversation). */
  readonly pendingId = signal<string | null>(null);
  /** Message currently being "typed out" and how much of it is visible. */
  readonly streamId = signal<string | null>(null);
  readonly streamText = signal('');

  private request?: Subscription;
  private streamTimer?: ReturnType<typeof setInterval>;
  private systemDark = signal(typeof matchMedia !== 'undefined' && matchMedia('(prefers-color-scheme: dark)').matches);

  // -------------------------------------------------------------- derived
  readonly active = computed(() => this.conversations().find((c) => c.id === this.activeId()));
  readonly messages = computed(() => this.active()?.messages ?? []);
  readonly run = computed(() => this.runs().find((r) => r.id === this.runId()) ?? null);
  readonly billing = computed(() => (this.run()?.kind ?? 'billing') === 'billing');
  readonly busy = computed(() => this.pendingId() !== null || this.streamId() !== null);
  readonly thinking = computed(() => this.pendingId() !== null && this.pendingId() === this.activeId());
  readonly isDark = computed(() => (this.theme() === 'system' ? this.systemDark() : this.theme() === 'dark'));
  readonly billingRuns = computed(() => this.runs().filter((r) => r.kind === 'billing'));
  readonly dailyRuns = computed(() => this.runs().filter((r) => r.kind === 'daily'));

  readonly suggestions = computed(() =>
    this.billing()
      ? [
          { title: 'Next billing cycle', sub: 'Forecast the upcoming two-month bill', q: 'Forecast next billing cycle' },
          { title: 'Recent trend', sub: 'Chart the last 6 billing periods', q: 'Graph last 6 periods' },
          { title: 'Averages', sub: 'Average total over the last 6 bills', q: 'Average last 6 periods' },
          { title: 'Model accuracy', sub: 'How it compares with simple baselines', q: 'How accurate is the model?' },
        ]
      : [
          { title: 'Tomorrow', sub: "Forecast tomorrow's supply", q: 'Forecast tomorrow' },
          { title: 'Next 7 days', sub: 'Chart a week-ahead forecast', q: 'Graph forecast 7 days' },
          { title: 'Averages', sub: 'Average over the last 30 days', q: 'Average last 30 days' },
          { title: 'Model accuracy', sub: 'How it compares with simple baselines', q: 'How accurate is the model?' },
        ],
  );

  readonly groups = computed(() => {
    const now = new Date();
    const today = new Date(now.getFullYear(), now.getMonth(), now.getDate()).getTime();
    const week = today - 6 * 86_400_000;
    const sorted = this.conversations().filter((c) => c.messages.length).sort((a, b) => b.updated - a.updated);
    const buckets: { label: string; items: Conversation[] }[] = [
      { label: 'Today', items: sorted.filter((c) => c.updated >= today) },
      { label: 'Previous 7 days', items: sorted.filter((c) => c.updated < today && c.updated >= week) },
      { label: 'Older', items: sorted.filter((c) => c.updated < week) },
    ];
    return buckets.filter((b) => b.items.length);
  });

  readonly bestNeural = computed(() => {
    const neural = (this.runDetail()?.metrics ?? []).filter((r) => r['type'] === 'neural');
    return neural.length ? [...neural].sort((a, b) => Number(a['mae']) - Number(b['mae']))[0] : null;
  });

  readonly greeting = computed(() => {
    const h = new Date().getHours();
    return h < 12 ? 'Good morning' : h < 17 ? 'Good afternoon' : 'Good evening';
  });

  constructor() {
    this.restorePrefs();
    this.restore();
    if (typeof matchMedia !== 'undefined') {
      matchMedia('(prefers-color-scheme: dark)').addEventListener('change', (e) => this.systemDark.set(e.matches));
    }
    effect(() => {
      const t = this.theme();
      document.documentElement.dataset['theme'] = t === 'system' ? (this.systemDark() ? 'dark' : 'light') : t;
    });
    effect(() => {
      const convs = this.conversations();
      try {
        localStorage.setItem(LS_KEY, JSON.stringify(convs.filter((c) => c.messages.length).slice(0, 40)));
      } catch { /* storage unavailable */ }
    });
    effect(() => {
      const prefs = { lang: this.lang(), theme: this.theme() };
      try {
        localStorage.setItem(LS_PREFS, JSON.stringify(prefs));
      } catch { /* storage unavailable */ }
    });
  }

  ngOnInit(): void {
    this.api.health().subscribe({
      next: () => {
        this.backendOk.set(true);
        this.loadRuns();
      },
      error: () => {
        this.backendOk.set(false);
        this.loadError.set('Backend not reachable at http://localhost:8000 — start it with: uvicorn api.main:app --port 8000');
        if (!this.active()) this.newChat();
      },
    });
  }

  // -------------------------------------------------------------- persistence
  private restorePrefs(): void {
    try {
      const p = JSON.parse(localStorage.getItem(LS_PREFS) ?? '{}');
      if (['auto', 'en', 'hinglish'].includes(p.lang)) this.lang.set(p.lang);
      if (['system', 'light', 'dark'].includes(p.theme)) this.theme.set(p.theme);
    } catch { /* defaults */ }
  }

  private restore(): void {
    try {
      const raw = localStorage.getItem(LS_KEY) ?? localStorage.getItem(LS_OLD_KEY);
      const parsed = raw ? (JSON.parse(raw) as Conversation[]) : [];
      if (!Array.isArray(parsed)) return;
      const convs = parsed.map((c) => ({
        ...c,
        updated: c.updated ?? Date.now(),
        // v1 had a canned greeting and string timestamps; drop the greeting, normalise the rest
        messages: (c.messages ?? [])
          .filter((m, i) => !(i === 0 && m.role === 'bot' && m.text.startsWith('Namaste! I answer')))
          .map((m) => ({ ...m, id: m.id ?? uid(), time: typeof m.time === 'number' ? m.time : Date.now() })),
      }));
      this.conversations.set(convs.filter((c) => c.messages.length));
    } catch { /* fresh start */ }
  }

  // -------------------------------------------------------------- runs
  private loadRuns(): void {
    this.api.runs().subscribe({
      next: (res) => {
        this.runs.set(res.runs);
        if (!res.runs.length) {
          this.loadError.set('No trained runs found in outputs/. Train once with run_experiment.py.');
          return;
        }
        const current = this.runId();
        if (!current || !res.runs.some((r) => r.id === current)) this.runId.set(res.default_run ?? res.runs[0].id);
        this.loadDetail();
        this.newChat();
      },
      error: () => this.loadError.set('Could not load model runs from the backend.'),
    });
  }

  private loadDetail(): void {
    const id = this.runId();
    if (!id) return;
    this.api.runDetail(id).subscribe({ next: (d) => this.runDetail.set(d) });
  }

  onRunChange(id: string): void {
    this.runId.set(id);
    this.loadDetail();
    this.newChat(); // a different model gets a fresh backend session
  }

  // -------------------------------------------------------------- conversations
  newChat(): void {
    this.finishStream();
    const empty = this.active();
    if (empty && !empty.messages.length) {
      this.patch(empty.id, (c) => ({ ...c, runId: this.runId(), sessionId: null }));
    } else {
      const c: Conversation = { id: uid(), title: 'New chat', sessionId: null, runId: this.runId(), messages: [], updated: Date.now() };
      this.conversations.update((list) => [c, ...list.filter((x) => x.messages.length)]);
      this.activeId.set(c.id);
    }
    this.closeSidebarOnMobile();
    this.focusBox();
  }

  select(id: string): void {
    this.finishStream();
    this.activeId.set(id);
    this.conversations.update((list) => list.filter((x) => x.messages.length || x.id === id));
    const c = this.active();
    if (c?.runId && c.runId !== this.runId() && this.runs().some((r) => r.id === c.runId)) {
      this.runId.set(c.runId);
      this.loadDetail();
    }
    this.closeSidebarOnMobile();
    this.scrollDown(true);
  }

  remove(id: string, ev: MouseEvent): void {
    ev.stopPropagation();
    if (this.pendingId() === id) this.stop();
    this.conversations.update((list) => list.filter((c) => c.id !== id));
    if (this.activeId() === id) {
      const next = this.conversations()[0];
      if (next) this.activeId.set(next.id);
      else this.newChat();
    }
  }

  private patch(id: string, fn: (c: Conversation) => Conversation): void {
    this.conversations.update((list) => list.map((c) => (c.id === id ? fn(c) : c)));
  }

  private push(id: string, msg: ChatMessage): void {
    this.patch(id, (c) => ({ ...c, messages: [...c.messages, msg], updated: Date.now() }));
  }

  // -------------------------------------------------------------- chat
  send(text?: string): void {
    const msg = (text ?? this.input()).trim();
    const conv = this.active();
    if (!msg || !conv || this.busy()) return;
    const convId = conv.id;
    this.input.set('');
    // the [value] binding may never have seen the typed text, so clear the element itself
    if (this.box) this.box.nativeElement.value = '';
    this.resizeBox();

    this.push(convId, { id: uid(), role: 'user', text: msg, time: Date.now() });
    if (conv.title === 'New chat') this.patch(convId, (c) => ({ ...c, title: msg.length > 48 ? msg.slice(0, 46) + '…' : msg }));
    this.scrollDown(true);

    const runId = this.runId();
    if (!this.backendOk() || !runId) {
      this.reply(convId, { id: uid(), role: 'bot', error: true, text: "I can't reach the forecasting backend right now. Start it with `uvicorn api.main:app --port 8000`, then try again.", time: Date.now() });
      return;
    }

    this.patch(convId, (c) => ({ ...c, runId }));
    this.pendingId.set(convId);
    this.request = this.api.chat(msg, runId, conv.sessionId, this.lang()).subscribe({
      next: (res) => {
        this.pendingId.set(null);
        this.patch(convId, (c) => ({ ...c, sessionId: res.session_id }));
        this.reply(convId, { id: uid(), role: 'bot', text: res.reply, chart: res.chart, forecast: res.forecast, topic: res.topic, time: Date.now() });
      },
      error: (err) => {
        this.pendingId.set(null);
        const detail = err?.error?.detail ?? (err?.status === 0 ? 'the backend is not responding' : err?.message ?? 'request failed');
        this.reply(convId, { id: uid(), role: 'bot', error: true, text: `Something went wrong — ${detail}. Please try again.`, time: Date.now() });
      },
    });
  }

  /** Add a bot message and type it out progressively, ChatGPT-style. */
  private reply(convId: string, msg: ChatMessage): void {
    this.push(convId, msg);
    if (convId !== this.activeId()) return;
    this.finishStream();
    const full = msg.text;
    const step = Math.max(2, Math.ceil(full.length / 90)); // whole reply in roughly 1.5 s
    let shown = 0;
    this.streamId.set(msg.id);
    this.streamText.set('');
    this.streamTimer = setInterval(() => {
      shown = Math.min(full.length, shown + step);
      // never cut a word or a **bold** marker in half
      let end = shown;
      while (end < full.length && !/\s/.test(full[end])) end++;
      this.streamText.set(full.slice(0, end));
      this.scrollDown();
      if (end >= full.length) this.finishStream();
    }, 16);
  }

  private finishStream(): void {
    if (this.streamTimer) clearInterval(this.streamTimer);
    this.streamTimer = undefined;
    if (this.streamId()) {
      this.streamId.set(null);
      this.scrollDown();
    }
  }

  stop(): void {
    if (this.request && this.pendingId()) {
      this.request.unsubscribe();
      const id = this.pendingId()!;
      this.pendingId.set(null);
      this.push(id, { id: uid(), role: 'bot', error: true, text: 'Stopped.', time: Date.now() });
    }
    this.finishStream();
    this.focusBox();
  }

  regenerate(): void {
    const msgs = this.messages();
    const lastUser = [...msgs].reverse().find((m) => m.role === 'user');
    const conv = this.active();
    if (!lastUser || !conv || this.busy()) return;
    const idx = msgs.lastIndexOf(lastUser);
    this.patch(conv.id, (c) => ({ ...c, messages: c.messages.slice(0, idx) }));
    this.send(lastUser.text);
  }

  async copy(m: ChatMessage): Promise<void> {
    try {
      await navigator.clipboard.writeText(m.text.replace(/\*\*/g, ''));
      this.copiedId.set(m.id);
      setTimeout(() => this.copiedId.set(null), 1500);
    } catch { /* clipboard blocked */ }
  }

  followUps(m: ChatMessage): string[] {
    const set = m.topic ? FOLLOW_UPS[m.topic] : undefined;
    return set ? (this.billing() ? set.billing : set.daily) : [];
  }

  // -------------------------------------------------------------- view helpers
  html(m: ChatMessage): string {
    if (this.streamId() !== m.id) return renderMarkdown(m.text);
    // mid-stream a **bold** span may be open; close it so raw asterisks never flash on screen
    let partial = this.streamText();
    if ((partial.match(/\*\*/g) ?? []).length % 2) partial += '**';
    return renderMarkdown(partial);
  }

  onInput(ev: Event): void {
    this.input.set((ev.target as HTMLTextAreaElement).value);
    this.resizeBox();
  }

  onKey(ev: KeyboardEvent): void {
    if (ev.key === 'Enter' && !ev.shiftKey && !ev.isComposing) {
      ev.preventDefault();
      this.send();
    }
  }

  cycleTheme(): void {
    this.theme.update((t) => (t === 'system' ? 'light' : t === 'light' ? 'dark' : 'system'));
  }

  fmt(v: number | string | null | undefined, digits = 2): string {
    if (v === null || v === undefined || Number.isNaN(Number(v))) return '–';
    return Number(v).toLocaleString('en-IN', { maximumFractionDigits: digits });
  }

  modelName(r: RunInfo): string {
    return r.variant_label ?? r.variant.replace(/_/g, ' ');
  }

  /** 'kharghar_20261010_212955' -> '10 Oct 2026, 9:29 PM' */
  runDate(id: string): string {
    const m = /(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})$/.exec(id);
    if (!m) return id;
    const d = new Date(+m[1], +m[2] - 1, +m[3], +m[4], +m[5]);
    return d.toLocaleString('en-IN', { day: 'numeric', month: 'short', year: 'numeric', hour: 'numeric', minute: '2-digit' });
  }

  clock(t: number): string {
    return new Date(t).toLocaleString([], { hour: '2-digit', minute: '2-digit', day: 'numeric', month: 'short' });
  }

  private resizeBox(): void {
    queueMicrotask(() => {
      const el = this.box?.nativeElement;
      if (!el) return;
      el.style.height = 'auto';
      el.style.height = Math.min(el.scrollHeight, 200) + 'px';
    });
  }

  private focusBox(): void {
    setTimeout(() => this.box?.nativeElement.focus(), 0);
  }

  private closeSidebarOnMobile(): void {
    if (window.innerWidth < 900) this.sidebarOpen.set(false);
  }

  /** Follow new output only when the user is already near the bottom (or when forced). */
  private scrollDown(force = false): void {
    requestAnimationFrame(() => {
      const el = this.thread?.nativeElement;
      if (!el) return;
      const near = el.scrollHeight - el.scrollTop - el.clientHeight < 160;
      if (force || near) el.scrollTop = el.scrollHeight;
    });
  }
}
