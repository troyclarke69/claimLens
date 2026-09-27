import { Routes } from '@angular/router';

/** Each page is lazy-loaded into its own chunk. */
export const routes: Routes = [
  { path: '', title: 'ClaimLens', loadComponent: () => import('./pages/overview').then((m) => m.Overview) },
  { path: 'demo', title: 'ClaimLens · Demo', loadComponent: () => import('./pages/demo/demo').then((m) => m.Demo) },
  { path: '**', redirectTo: '' },
];
