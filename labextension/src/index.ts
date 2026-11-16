import {
  JupyterFrontEnd,
  JupyterFrontEndPlugin
} from '@jupyterlab/application';
import { INotebookTracker, NotebookPanel } from '@jupyterlab/notebook';
import { NotebookActions } from '@jupyterlab/notebook';
import { Dialog, showDialog, ToolbarButton } from '@jupyterlab/apputils';
import { LabIcon, refreshIcon } from '@jupyterlab/ui-components';

const GITHUB_REPO = 'duoan/TorchCode';
const GITHUB_BRANCH = 'master';
const TEMPLATES_BACKUP = '_original_templates';

const colabIcon = new LabIcon({
  name: 'torchcode:colab',
  svgstr: `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">
    <path fill="#E8710A" d="M19 19H5V5h7V3H5a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7h-2v7zM14 3v2h3.59l-9.83 9.83 1.41 1.41L19 6.41V10h2V3h-7z"/>
  </svg>`
});

const runTestsIcon = new LabIcon({
  name: 'torchcode:run-tests',
  svgstr: `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
    <path d="M9 11l3 3L22 4"/><path d="M21 12v7a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11"/>
  </svg>`
});

const hintIcon = new LabIcon({
  name: 'torchcode:hint',
  svgstr: `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
    <circle cx="12" cy="12" r="10"/><path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3"/><path d="M12 17h.01"/>
  </svg>`
});

const statusIcon = new LabIcon({
  name: 'torchcode:status',
  svgstr: `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
    <path d="M3 3v18h18"/><path d="M18 9V5"/><path d="M12 9V3"/><path d="M6 9V2"/>
  </svg>`
});

function getFilename(path: string): string {
  return path.split('/').pop() || '';
}

/**
 * Notebook file stems whose judge task id differs.
 * Keep in sync with TASK_ID_ALIASES in api/parser.py.
 */
const TASK_ID_ALIASES: Record<string, string> = {
  multihead_attention: 'mha'
};

/**
 * Infer task_id from notebook path, e.g. 01_relu.ipynb -> relu,
 * 39_ppo_loss_solution.ipynb -> ppo_loss, 06_multihead_attention.ipynb -> mha.
 */
function getTaskIdFromPath(path: string): string | null {
  const base = getFilename(path);
  if (!base.endsWith('.ipynb')) return null;
  const name = base.replace(/\.ipynb$/i, '');
  if (name === '00_welcome') return null;
  const stem = name.replace(/^\d+_/, '').replace(/_solution$/, '');
  if (!stem) return null;
  return TASK_ID_ALIASES[stem] ?? stem;
}

function getGitHubDir(filename: string): string {
  return filename.includes('_solution') ? 'solutions' : 'templates';
}

function getColabUrl(filename: string): string {
  const dir = getGitHubDir(filename);
  return `https://colab.research.google.com/github/${GITHUB_REPO}/blob/${GITHUB_BRANCH}/${dir}/${filename}`;
}

/**
 * Find an existing code cell whose source contains `pattern` (substring match).
 * Returns the cell index, or -1 if not found.
 */
function findCellByContent(panel: NotebookPanel, pattern: string): number {
  const cellCount = panel.content.model?.cells.length ?? 0;
  for (let i = 0; i < cellCount; i++) {
    const cellModel = panel.content.model!.cells.get(i);
    if (cellModel.type === 'code' && cellModel.sharedModel.getSource().includes(pattern)) {
      return i;
    }
  }
  return -1;
}

/**
 * Find an existing cell matching `pattern`, select it, and run it.
 * If no matching cell exists, insert one at the end with `code` and run it.
 */
async function runExistingOrInsert(panel: NotebookPanel, code: string, pattern: string): Promise<void> {
  const notebook = panel.content;
  const sessionContext = panel.sessionContext;
  if (!sessionContext.session?.kernel) {
    await showDialog({
      title: 'No kernel',
      body: 'Start a kernel first (e.g. Run > Run All Cells or run any cell).',
      buttons: [Dialog.okButton()]
    });
    return;
  }

  const idx = findCellByContent(panel, pattern);
  if (idx >= 0) {
    notebook.activeCellIndex = idx;
  } else {
    const cellCount = notebook.model?.cells.length ?? 0;
    notebook.activeCellIndex = cellCount - 1;
    NotebookActions.insertBelow(notebook);
    const cell = notebook.activeCell;
    if (cell && cell.model.type === 'code') {
      cell.model.sharedModel.setSource(code);
    }
  }

  await NotebookActions.run(notebook, sessionContext);
}

const COMMAND_RESET = 'torchcode:reset-notebook';
const COMMAND_COLAB = 'torchcode:open-in-colab';
const COMMAND_RUN_TESTS = 'torchcode:run-tests';
const COMMAND_HINT = 'torchcode:get-hint';
const COMMAND_STATUS = 'torchcode:show-status';

const plugin: JupyterFrontEndPlugin<void> = {
  id: 'torchcode-labext:plugin',
  autoStart: true,
  requires: [INotebookTracker],
  activate: (app: JupyterFrontEnd, tracker: INotebookTracker) => {
    console.log('TorchCode extension activated');

    // Inject Google Fonts into the page
    if (!document.querySelector('link[data-torchcode-fonts]')) {
      const link = document.createElement('link');
      link.rel = 'stylesheet';
      link.href = 'https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap';
      link.setAttribute('data-torchcode-fonts', 'true');
      document.head.appendChild(link);
    }

    app.commands.addCommand(COMMAND_RESET, {
      label: 'Reset to Template',
      caption: 'Reset this notebook to its original template state',
      icon: refreshIcon,
      execute: async () => {
        const current = tracker.currentWidget;
        if (!current) return;

        const path = current.context.path;
        const filename = getFilename(path);
        if (!filename || filename === '00_welcome.ipynb') return;

        const result = await showDialog({
          title: '🔄 Reset Notebook',
          body: `Reset "${filename}" to template state?\nAll your changes will be lost.`,
          buttons: [
            Dialog.cancelButton(),
            Dialog.warnButton({ label: 'Reset' })
          ]
        });

        if (!result.button.accept) return;

        try {
          const templatePath = `${TEMPLATES_BACKUP}/${filename}`;
          const template = await app.serviceManager.contents.get(templatePath, {
            content: true,
            type: 'notebook'
          });

          await app.serviceManager.contents.save(path, {
            type: 'notebook',
            format: 'json',
            content: template.content
          });

          await current.context.revert();
        } catch (err) {
          console.error('Reset failed:', err);
          await showDialog({
            title: 'Reset Failed',
            body: `Could not find template for "${filename}".\nMake sure _original_templates/ exists.`,
            buttons: [Dialog.okButton()]
          });
        }
      }
    });

    app.commands.addCommand(COMMAND_COLAB, {
      label: 'Open in Colab',
      caption: 'Open this notebook in Google Colab',
      icon: colabIcon,
      execute: () => {
        const current = tracker.currentWidget;
        if (!current) return;

        const filename = getFilename(current.context.path);
        if (!filename) return;

        window.open(getColabUrl(filename), '_blank');
      }
    });

    app.commands.addCommand(COMMAND_RUN_TESTS, {
      label: 'Run Tests',
      caption: 'Run judge tests for this problem',
      icon: runTestsIcon,
      execute: async () => {
        const current = tracker.currentWidget;
        if (!current) return;

        const taskId = getTaskIdFromPath(current.context.path);
        if (!taskId) {
          await showDialog({
            title: 'Not a problem notebook',
            body: 'Open a problem template (e.g. 01_relu.ipynb) to run tests.',
            buttons: [Dialog.okButton()]
          });
          return;
        }
        await runExistingOrInsert(
          current,
          `from torch_judge import check\ncheck("${taskId}")`,
          `check("${taskId}")`
        );
      }
    });

    app.commands.addCommand(COMMAND_HINT, {
      label: 'Get Hint',
      caption: 'Show a hint for this problem',
      icon: hintIcon,
      execute: async () => {
        const current = tracker.currentWidget;
        if (!current) return;

        const taskId = getTaskIdFromPath(current.context.path);
        if (!taskId) {
          await showDialog({
            title: 'Not a problem notebook',
            body: 'Open a problem template (e.g. 01_relu.ipynb) to get a hint.',
            buttons: [Dialog.okButton()]
          });
          return;
        }
        await runExistingOrInsert(
          current,
          `from torch_judge import hint\nhint("${taskId}")`,
          `hint("${taskId}")`
        );
      }
    });

    app.commands.addCommand(COMMAND_STATUS, {
      label: 'Show Progress',
      caption: 'Show solved / attempted / todo status',
      icon: statusIcon,
      execute: async () => {
        const current = tracker.currentWidget;
        if (!current) return;
        await runExistingOrInsert(
          current,
          'from torch_judge import status\nstatus()',
          'status()'
        );
      }
    });

    // Keyboard shortcuts: Accel+Shift+T Run Tests, Accel+Shift+H Hint (Accel = Cmd on Mac, Ctrl on Win/Linux)
    app.commands.addKeyBinding({
      command: COMMAND_RUN_TESTS,
      keys: ['Accel Shift T'],
      selector: '.jp-Notebook'
    });
    app.commands.addKeyBinding({
      command: COMMAND_HINT,
      keys: ['Accel Shift H'],
      selector: '.jp-Notebook'
    });

    tracker.widgetAdded.connect(
      (_sender: INotebookTracker, panel: NotebookPanel) => {
        const runTestsButton = new ToolbarButton({
          icon: runTestsIcon,
          tooltip: 'Run judge tests (Ctrl+Shift+T)',
          label: 'Run Tests',
          onClick: () => app.commands.execute(COMMAND_RUN_TESTS),
          className: 'torchcode-runtests-button'
        });
        panel.toolbar.addItem('torchcode-runtests', runTestsButton);

        const hintButton = new ToolbarButton({
          icon: hintIcon,
          tooltip: 'Get hint (Ctrl+Shift+H)',
          label: 'Hint',
          onClick: () => app.commands.execute(COMMAND_HINT),
          className: 'torchcode-hint-button'
        });
        panel.toolbar.addItem('torchcode-hint', hintButton);

        const statusButton = new ToolbarButton({
          icon: statusIcon,
          tooltip: 'Show progress',
          label: 'Status',
          onClick: () => app.commands.execute(COMMAND_STATUS),
          className: 'torchcode-status-button'
        });
        panel.toolbar.addItem('torchcode-status', statusButton);

        const colabButton = new ToolbarButton({
          icon: colabIcon,
          tooltip: 'Open in Google Colab',
          label: 'Colab',
          onClick: () => app.commands.execute(COMMAND_COLAB),
          className: 'torchcode-colab-button'
        });
        panel.toolbar.addItem('torchcode-colab', colabButton);

        const resetButton = new ToolbarButton({
          icon: refreshIcon,
          tooltip: 'Reset notebook to template state',
          label: 'Reset',
          onClick: () => app.commands.execute(COMMAND_RESET),
          className: 'torchcode-reset-button'
        });
        panel.toolbar.addItem('torchcode-reset', resetButton);
      }
    );
  }
};

export default plugin;
