import { Injectable, inject } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

export interface RunInfo {
  id: string;
  kind: 'billing' | 'daily';
  frequency: string;
  variant: string;
  variant_label?: string;
  sequence_length: number;
  history_rows: number | null;
  last_label: string | null;
  last_value: number | null;
  evaluation: Record<string, unknown>;
}

export interface ForecastPoints {
  labels: string[];
  values: number[];
  lower?: number[] | null;
  upper?: number[] | null;
  last_label?: string;
  last_value?: number;
  unit: string;
}

export interface ChartDataset {
  label: string;
  data: (number | null)[];
}

export interface ChartPayload {
  kind: 'line' | 'bar' | 'forecast';
  title: string;
  x_label: string;
  y_label: string;
  labels: string[];
  datasets: ChartDataset[];
  forecast?: ForecastPoints;
}

export interface ChatMessage {
  id: string;
  role: 'user' | 'bot';
  text: string;
  chart?: ChartPayload | null;
  forecast?: ForecastPoints | null;
  topic?: string | null;
  error?: boolean;
  time: number;
}

export interface RunDetail {
  run: RunInfo;
  metrics: Record<string, number | string>[];
  recent: { labels: string[]; values: number[] };
}

export interface ChatReply {
  reply: string;
  run_id: string;
  session_id: string;
  lang: string;
  topic: string | null;
  chart: ChartPayload | null;
  forecast: ForecastPoints | null;
  history: { labels: string[]; values: number[] } | null;
}

const API = 'http://localhost:8000/api';

@Injectable({ providedIn: 'root' })
export class ApiService {
  private http = inject(HttpClient);

  health(): Observable<{ status: string; runs: number }> {
    return this.http.get<{ status: string; runs: number }>(`${API}/health`);
  }

  runs(): Observable<{ runs: RunInfo[]; default_run: string | null }> {
    return this.http.get<{ runs: RunInfo[]; default_run: string | null }>(`${API}/runs`);
  }

  runDetail(runId: string): Observable<RunDetail> {
    return this.http.get<RunDetail>(`${API}/runs/${runId}`);
  }

  chat(message: string, runId: string | null, sessionId: string | null, lang: string): Observable<ChatReply> {
    return this.http.post<ChatReply>(`${API}/chat`, {
      message,
      run_id: runId,
      session_id: sessionId,
      lang,
    });
  }
}
