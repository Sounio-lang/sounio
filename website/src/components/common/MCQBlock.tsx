import { useState } from 'react';

export type MCQ = {
  question: string;
  options: string[];
  correct: number;
  explanation: string;
};

function QuestionItem({ q, selected, onSelect }: {
  q: MCQ;
  selected: number | null;
  onSelect: (i: number) => void;
}) {
  const answered = selected !== null;
  const isCorrect = selected === q.correct;

  return (
    <div className="grid gap-2">
      <p className="text-[0.85rem] text-[var(--color-text-primary)] leading-[1.6] font-[500]">
        {q.question}
      </p>
      <div className="grid gap-1.5">
        {q.options.map((opt, i) => {
          let style = 'border border-[var(--glass-border)] bg-[rgba(255,255,255,0.03)]';
          let label = '';
          if (answered) {
            if (i === q.correct) {
              style = 'border border-emerald-500/60 bg-emerald-500/10';
              label = ' ✓';
            } else if (i === selected) {
              style = 'border border-red-500/60 bg-red-500/10';
              label = ' ✗';
            } else {
              style = 'border border-[var(--glass-border)] bg-transparent opacity-50';
            }
          }
          return (
            <button
              key={i}
              disabled={answered}
              onClick={() => onSelect(i)}
              className={`w-full text-left px-3 py-2 rounded-lg text-[0.82rem] text-[var(--color-text-secondary)] transition-all leading-[1.5] ${style} ${!answered ? 'hover:bg-[rgba(255,255,255,0.07)] hover:text-[var(--color-text-primary)] cursor-pointer' : 'cursor-default'}`}
            >
              <span className="font-mono text-[var(--color-text-tertiary)] mr-2">
                {String.fromCharCode(65 + i)}.
              </span>
              {opt}{label}
            </button>
          );
        })}
      </div>
      {answered && (
        <p className={`text-[0.78rem] leading-[1.55] px-3 py-2 rounded-lg ${isCorrect ? 'bg-emerald-500/10 text-emerald-300' : 'bg-red-500/10 text-red-300'}`}>
          {q.explanation}
        </p>
      )}
    </div>
  );
}

export function MCQBlock({ questions }: { questions: MCQ[] }) {
  const [answers, setAnswers] = useState<(number | null)[]>(questions.map(() => null));

  return (
    <div className="grid gap-5 mt-6 pt-6 border-t border-[var(--glass-border)]">
      <p className="text-[0.72rem] font-[700] uppercase tracking-[0.12em] text-[var(--color-text-tertiary)]">
        Test your understanding
      </p>
      {questions.map((q, qi) => (
        <QuestionItem
          key={qi}
          q={q}
          selected={answers[qi]}
          onSelect={(i) => setAnswers(prev => { const n = [...prev]; n[qi] = i; return n; })}
        />
      ))}
    </div>
  );
}
