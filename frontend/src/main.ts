import { bootstrapApplication } from '@angular/platform-browser';
import { appConfig } from './app/app.config';
import { Root } from './app/dash/shell';

bootstrapApplication(Root, appConfig)
  .catch((err) => console.error(err));
