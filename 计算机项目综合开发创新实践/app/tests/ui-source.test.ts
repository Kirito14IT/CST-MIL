import assert from 'node:assert/strict';
import test from 'node:test';
import { sourceParts } from '../src/TopicCard.tsx';

test('original codepoint spans preserve emoji and composed/decomposed accents', () => {
  const text = '🙂 café / e\u0301';
  assert.deepEqual(sourceParts(text, { messageId: 'u', start: 2, end: 6, excerpt: 'café' }), { before: '🙂 ', match: 'café', after: ' / e\u0301' });
  assert.deepEqual(sourceParts(text, { messageId: 'u', start: 9, end: 11, excerpt: 'e\u0301' }), { before: '🙂 café / ', match: 'e\u0301', after: '' });
});

test('mismatched, UTF16, negative and out-of-range spans never highlight another excerpt', () => {
  const text = '🙂 café / e\u0301';
  for (const span of [
    { start: 3, end: 7, excerpt: 'café' },
    { start: 9, end: 11, excerpt: 'é' },
    { start: -1, end: 2, excerpt: '🙂 ' },
    { start: 9, end: 12, excerpt: 'e\u0301' },
  ]) assert.equal(sourceParts(text, { messageId: 'u', ...span }), null);
});
