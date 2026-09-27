/** Plain-language names for the evaluator's error taxonomy (claimlens/evaluation.py). */

export type Tone = 'good' | 'warn' | 'bad' | 'muted';

export const VERDICTS: Record<string, { label: string; tone: Tone; help: string }> = {
  correct: { label: 'Correct', tone: 'good', help: 'Value matches the ground truth.' },
  correct_null: { label: 'Correctly absent', tone: 'good', help: 'Not on the page, and the model said so.' },
  near_miss: { label: 'Near miss', tone: 'warn', help: 'Off by a character or two: a reading error.' },
  diacritics: { label: 'Accent dropped', tone: 'warn', help: 'Right apart from accents (é → e).' },
  missed: { label: 'Missed', tone: 'warn', help: 'On the page, but the model returned null.' },
  distractor_confusion: { label: 'Wrong value on page', tone: 'bad', help: 'Picked a look-alike value (e.g. date of birth).' },
  distractor_for_absent: { label: 'Filled from a look-alike', tone: 'bad', help: 'Field is absent, but the model copied a similar value.' },
  field_swap: { label: 'Field swap', tone: 'bad', help: "Returned another field's value." },
  hallucinated: { label: 'Invented', tone: 'bad', help: 'Field is absent, and the value is not on the page.' },
  wrong_value: { label: 'Wrong', tone: 'bad', help: 'Value does not match.' },
  unparseable: { label: 'Unparseable', tone: 'bad', help: 'The model output was not valid JSON.' },
};

export function verdict(errorType: string | undefined) {
  return VERDICTS[errorType ?? ''] ?? { label: errorType ?? '—', tone: 'muted' as Tone, help: '' };
}
