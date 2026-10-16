import { Component, ElementRef, Input, OnChanges, OnDestroy, ViewChild } from '@angular/core';
import { Chart, ChartConfiguration, registerables } from 'chart.js';
import type { ChartPayload } from './api.service';

Chart.register(...registerables);

const PALETTE = {
  light: { line: '#0b7a8f', fill: 'rgba(11,122,143,.10)', fc: '#d0612f', fcFill: 'rgba(208,97,47,.10)', best: '#0b7a8f', bar: '#c9c8c2', text: '#5f5f5a', grid: 'rgba(0,0,0,.06)', tip: '#1d1d1b' },
  dark: { line: '#3fb6c9', fill: 'rgba(63,182,201,.12)', fc: '#f08a5d', fcFill: 'rgba(240,138,93,.12)', best: '#3fb6c9', bar: '#4a4a46', text: '#b4b4ae', grid: 'rgba(255,255,255,.07)', tip: '#ececea' },
};

const compact = (v: number | string) => {
  const n = Number(v);
  if (Math.abs(n) >= 1e7) return (n / 1e7).toFixed(1) + ' Cr';
  if (Math.abs(n) >= 1e5) return (n / 1e5).toFixed(1) + ' L';
  if (Math.abs(n) >= 1e3) return (n / 1e3).toFixed(1) + 'k';
  return String(n);
};

/** Renders backend chart JSON (line / bar / history+forecast) with Chart.js. */
@Component({
  selector: 'app-chart',
  standalone: true,
  template: `<div class="chart-title">{{ chart.title }}</div><div class="chart-wrap"><canvas #cv></canvas></div>`,
  styles: [
    `.chart-title{font-size:13.5px;font-weight:500;color:var(--text-2);margin:0 0 8px 4px}
     .chart-wrap{position:relative;height:260px;width:100%}
     canvas{width:100%!important;height:100%!important}`,
  ],
})
export class ChartComponent implements OnChanges, OnDestroy {
  @Input() chart!: ChartPayload;
  @Input() dark = false;
  @ViewChild('cv', { static: true }) cv!: ElementRef<HTMLCanvasElement>;
  private inst: Chart | null = null;

  ngOnChanges(): void {
    this.draw();
  }

  ngOnDestroy(): void {
    this.inst?.destroy();
  }

  private draw(): void {
    if (!this.chart) return;
    this.inst?.destroy();
    const c = this.chart;
    const p = this.dark ? PALETTE.dark : PALETTE.light;
    const font = { family: 'Inter, system-ui, sans-serif', size: 12 };
    const tooltip = {
      backgroundColor: this.dark ? '#262625' : '#ffffff',
      titleColor: p.tip,
      bodyColor: p.tip,
      borderColor: this.dark ? '#42423e' : '#e6e5e1',
      borderWidth: 1,
      padding: 10,
      cornerRadius: 8,
      titleFont: { ...font, weight: 600 as const },
      bodyFont: font,
      callbacks: {
        label: (ctx: { dataset: { label?: string }; parsed: { y: number | null } }) => {
          if (ctx.parsed.y == null) return '';
          const name = ctx.dataset.label === 'Likely range' ? 'Range high' : ctx.dataset.label;
          return ` ${name}: ${ctx.parsed.y.toLocaleString('en-IN', { maximumFractionDigits: 0 })}`;
        },
      },
    };
    const axis = (title: string) => ({
      title: { display: !!title, text: title, color: p.text, font },
      ticks: { color: p.text, font, callback: compact as never },
      grid: { color: p.grid },
      border: { display: false },
    });

    if (c.kind === 'bar') {
      const cfg: ChartConfiguration<'bar'> = {
        type: 'bar',
        data: {
          labels: c.labels,
          datasets: [{ label: c.datasets[0]?.label ?? 'RMSE', data: c.datasets[0]?.data ?? [], backgroundColor: c.labels.map((_, i) => (i === 0 ? p.best : p.bar)), borderRadius: 6, maxBarThickness: 48 }],
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: { legend: { display: false }, tooltip: tooltip as never },
          scales: { x: { ticks: { color: p.text, font }, grid: { display: false }, border: { display: false } }, y: axis(c.y_label) as never },
        },
      };
      this.inst = new Chart(this.cv.nativeElement, cfg);
      return;
    }

    const datasets = c.datasets.map((d) => {
      const label = d.label.toLowerCase();
      if (label === 'range low' || label === 'likely range') {
        // invisible edges; the upper edge fills down to the lower one ('-1' = previous dataset)
        return {
          label: d.label,
          data: d.data,
          borderColor: 'transparent',
          backgroundColor: p.fcFill,
          pointBackgroundColor: p.fc + '55', // legend swatch only; points are hidden
          pointBorderColor: 'transparent',
          pointRadius: 0,
          pointHoverRadius: 0,
          tension: 0.3,
          spanGaps: true,
          fill: label === 'likely range' ? '-1' : false,
        };
      }
      const fc = label.includes('forecast');
      return {
        label: d.label,
        data: d.data,
        borderColor: fc ? p.fc : p.line,
        backgroundColor: fc ? p.fcFill : p.fill,
        pointBackgroundColor: fc ? p.fc : p.line,
        borderWidth: 2,
        pointRadius: 3,
        pointHoverRadius: 5,
        tension: 0.3,
        spanGaps: true,
        fill: !fc,
        borderDash: fc ? [6, 4] : undefined,
      };
    });
    const cfg: ChartConfiguration<'line'> = {
      type: 'line',
      data: { labels: c.labels, datasets: datasets as never },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: { mode: 'index', intersect: false },
        plugins: {
          legend: {
            display: datasets.length > 1,
            position: 'bottom',
            labels: { color: p.text, font, usePointStyle: true, pointStyle: 'circle', boxWidth: 6, boxHeight: 6, filter: (item) => item.text !== 'Range low' },
          },
          tooltip: tooltip as never,
        },
        scales: {
          x: { ticks: { color: p.text, font, maxRotation: 0, autoSkip: true, maxTicksLimit: 8 }, grid: { display: false }, border: { display: false } },
          y: axis(c.y_label) as never,
        },
      },
    };
    this.inst = new Chart(this.cv.nativeElement, cfg);
  }
}
