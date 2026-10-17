export const scientistLanguages = ['en', 'pt', 'el'];
export const scientistFields = ['question', 'data', 'units', 'assumptions', 'deliverables', 'validation'];
export const scientistFieldLimit = 3000;

export function validateScientistCatalog(catalog, taskIds) {
  const errors = [];
  if (catalog?.schema !== 'sounio.scientist-workflows.v1') errors.push('invalid workflow schema');
  if (!/^[a-f0-9]{40}$/.test(catalog?.source_revision ?? '')) errors.push('invalid source revision');
  if (!Array.isArray(catalog?.workflows) || !catalog.workflows.length) return [...errors, 'missing workflows'];
  const identifiers = new Set();
  for (const workflow of catalog.workflows) {
    if (!workflow || !/^[a-z][a-z0-9-]*$/.test(workflow.id ?? '') || identifiers.has(workflow.id)) { errors.push('invalid or duplicate workflow'); continue; }
    identifiers.add(workflow.id);
    if (!['chemistry', 'biomedicine', 'ai'].includes(workflow.category)) errors.push(`${workflow.id}: invalid category`);
    if (workflow.state !== 'prompt-template-not-executed') errors.push(`${workflow.id}: unsupported execution claim`);
    if (!taskIds.includes(workflow.starter_task)) errors.push(`${workflow.id}: unknown language task`);
    if (!Array.isArray(workflow.sources) || !workflow.sources.length || workflow.sources.some(source => typeof source !== 'string' || !/^(?:examples|tests|demos|stdlib)\/[a-zA-Z0-9_./-]+\.sio$/.test(source) || source.split('/').some(part => !part || part === '.' || part === '..'))) errors.push(`${workflow.id}: unsafe source path`);
    for (const language of scientistLanguages) {
      for (const field of ['title', 'goal']) if (typeof workflow[field]?.[language] !== 'string' || !workflow[field][language].trim()) errors.push(`${workflow.id}: missing ${language} ${field}`);
      for (const field of ['questions', 'acceptance', 'limits']) if (!Array.isArray(workflow[field]?.[language]) || workflow[field][language].length < 2 || workflow[field][language].some(value => typeof value !== 'string' || !value.trim())) errors.push(`${workflow.id}: missing ${language} ${field}`);
    }
  }
  return errors;
}

export function buildScientistPrompt({ workflow, language, copy, answers, context }) {
  if (!scientistLanguages.includes(language)) throw new Error('Unsupported scientist language');
  if (!workflow || workflow.state !== 'prompt-template-not-executed' || typeof context?.language_pack !== 'string' || !context.language_pack.trim() || !Array.isArray(context?.sources) || !context.sources.length || context.sources.some(source => typeof source.path !== 'string' || !/^[a-f0-9]{64}$/.test(source.sha256 ?? '') || !/^[a-f0-9]{40}$/.test(source.revision ?? '') || source.url !== `https://github.com/Sounio-lang/sounio/blob/${source.revision}/${source.path}`)) throw new Error('Missing source-bound workflow context');
  const brief = {};
  for (const field of scientistFields) {
    const value = answers[field] ?? '';
    if (typeof value !== 'string' || value.length > scientistFieldLimit) throw new Error('Invalid scientist field');
    brief[field] = value.trim() || copy.unknown;
  }
  if (!answers.question?.trim()) throw new Error('Missing scientific question');
  const delimiter = '`'.repeat(Math.max(3, ...(JSON.stringify(brief).match(/`+/g) ?? []).map(run => run.length + 1)));
  const lines = [
    `# ${workflow.title[language]}`, '', `Language: ${language}`, `Workflow: ${workflow.id}`, `State: ${workflow.state}`, 'First reply phase: INTAKE_ONLY', '',
    `## ${copy.intakeTitle}`, ...copy.intake.map(instruction => `- ${instruction}`), '',
    `## ${copy.contractTitle}`, ...copy.contract.map((instruction, index) => `${index + 1}. ${instruction}`), '',
    `## ${copy.recipeIntent}`, workflow.goal[language], '', `## ${copy.brief}`, `${delimiter}json`, JSON.stringify(brief, null, 2), delimiter, '',
    `## ${copy.questions}`, ...workflow.questions[language].map(question => `- ${question}`), '',
    `## ${copy.acceptance}`, ...workflow.acceptance[language].map(criterion => `- ${criterion}`), '',
    `## ${copy.limits}`, ...workflow.limits[language].map(limit => `- ${limit}`), '',
    `## ${copy.sources}`, ...context.sources.map(source => `- ${source.path}; SHA-256=${source.sha256}; revision=${source.revision}; source-identity-pinned; model-access-not-established; not-executed\n  ${source.url}`), '',
    context.language_pack, '', `## ${copy.intakeTitle}`, ...copy.intake.map(instruction => `- ${instruction}`),
  ];
  return lines.join('\n');
}
