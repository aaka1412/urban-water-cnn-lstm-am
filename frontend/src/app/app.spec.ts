import { TestBed } from '@angular/core/testing';
import { provideHttpClient } from '@angular/common/http';
import { App } from './app';
import { renderMarkdown } from './markdown';

describe('App', () => {
  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [App],
      providers: [provideHttpClient()],
    }).compileComponents();
  });

  it('should create the app', () => {
    const fixture = TestBed.createComponent(App);
    expect(fixture.componentInstance).toBeTruthy();
  });
});

describe('renderMarkdown', () => {
  it('escapes html and renders bold and lists', () => {
    const html = renderMarkdown(['<b>x</b> **bold**', '- one', '- two'].join('\n'));
    expect(html).toContain('&lt;b&gt;');
    expect(html).toContain('<strong>bold</strong>');
    expect(html).toContain('<ul><li>one</li><li>two</li></ul>');
  });
});
