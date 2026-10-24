import { Routes } from '@angular/router';
import { adminGuard, authGuard } from './core';
import { Login, Shell } from './shell';

export const dashRoutes: Routes = [
  { path: 'login', component: Login },
  { path: 'admin/login', component: Login },
  {
    path: '', component: Shell, canActivate: [authGuard],
    children: [
      { path: '', pathMatch: 'full', redirectTo: 'overview' },
      { path: 'overview', loadComponent: () => import('./pages-user').then((m) => m.Overview) },
      { path: 'upload', loadComponent: () => import('./pages-user').then((m) => m.Upload) },
      { path: 'analytics', loadComponent: () => import('./pages-user').then((m) => m.Analytics) },
      { path: 'predict', loadComponent: () => import('./pages-user').then((m) => m.Predict) },
      { path: 'models', loadComponent: () => import('./pages-eval').then((m) => m.Compare) },
      { path: 'training', loadComponent: () => import('./pages-eval').then((m) => m.Training) },
      { path: 'confusion', loadComponent: () => import('./pages-eval').then((m) => m.Confusion) },
      { path: 'history', loadComponent: () => import('./pages-user').then((m) => m.History) },
      { path: 'reports', loadComponent: () => import('./pages-user').then((m) => m.Reports) },
      { path: 'profile', loadComponent: () => import('./pages-user').then((m) => m.Profile) },
      { path: 'admin', canActivate: [adminGuard], loadComponent: () => import('./pages-admin').then((m) => m.Admin) },
    ],
  },
  { path: 'chatbot', canActivate: [authGuard], loadComponent: () => import('../app').then((m) => m.App) },
  { path: '**', redirectTo: '' },
];
