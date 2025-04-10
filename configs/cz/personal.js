module.exports = {
  types: [
    { value: 'Enh', name: 'Enh:      A new feature or enhancement' },
    { value: 'Fix', name: 'Fix:      A bug fix' },
    { value: 'Docs', name: 'Docs:     Documentation only changes' },
    {
      value: 'Style',
      name: 'Style:    Changes that do not affect the meaning of the code\n            (white-space, formatting, missing semi-colons, etc)',
    },
    {
      value: 'Refactor',
      name: 'Refactor: A code change that neither fixes a bug nor adds a feature',
    },
    {
      value: 'Perf',
      name: 'Perf:     A code change that improves performance',
    },
    { value: 'Test', name: 'Test:     Adding missing tests' },
    {
      value: 'Build',
      name: 'Build:    Changes to the build process',
    },
    {
      value: 'Chore',
      name: 'Chore:    Changes to the auxiliary tools (such as linters)\n            and libraries such as documentation generation',
    },
    { value: 'Revert', name: 'Revert:   Revert to a commit' },
    { value: 'Wip', name: 'Wip:      Work in progress' },
    { value: 'Sync', name: 'Sync:     Sync changes from submodules / libraries' },
  ],

  scopes: [{ name: 'Ci' }, { name: 'Make' }],

  typePrefix: "[",
  typeSuffix: "]",

  allowTicketNumber: false,
  isTicketNumberRequired: false,
  ticketNumberPrefix: 'TICKET-',
  ticketNumberSuffix: '',
  ticketNumberRegExp: '\\d{1,5}',

  // It needs to match the value for field type. Eg.: 'fix'
  scopeOverrides: {
    Fix: [
      { name: 'Merge' },
      { name: 'E2eTest' },
      { name: 'UnitTest' }
    ],
    Docs: [
      { name: 'Description' },
      { name: 'Contribution' },
      { name: 'License' },
    ],
    Chore: [
      { name: 'Linter' },
      { name: 'Formatter' },
      { name: 'PreCommit' },
      { name: 'Commitizen' },
      { name: 'Gitignore' },
    ],
  },

  // override the messages, defaults are as follows
  messages: {
    type: "Select the type of change that you're committing:",
    scope: '\nDenote the SCOPE of this change (optional):',
    // used if allowCustomScopes is true
    customScope: 'Denote the SCOPE of this change:',
    subject: 'Write a SHORT, PAST tense description of the change with dot at the end:\n',
    body: 'Provide a LONGER description of the change (optional). Use "|" to break new line:\n',
    breaking: 'List any BREAKING CHANGES (optional):\n',
    footer: 'List any ISSUES CLOSED by this change (optional). E.g.: #31, #34:\n',
    confirmCommit: 'Are you sure you want to proceed with the commit above?',
  },

  allowCustomScopes: true,
  allowBreakingChanges: ['feat', 'fix'],
  // skip any questions you want
  skipQuestions: ['body'],

  // Subject (description of the commit).
  // - Limit subject length.
  subjectLimit: 100,
  // - Always start subject from capital letter.
  upperCaseSubject: true,
  // breaklineChar: '|', // It is supported for fields body and footer.
  // footerPrefix : 'ISSUES CLOSED:'
  // askForBreakingChangeFirst : true, // default is false
};
