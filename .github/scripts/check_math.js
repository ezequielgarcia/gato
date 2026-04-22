#!/usr/bin/env node
/**
 * Validate every LaTeX math block in the repository's Markdown files.
 *
 * Extracts $$...$$ (display math) and $...$ (inline math) blocks from
 * every tracked .md file and runs each one through KaTeX — the same
 * renderer GitHub uses for math in README and Markdown previews. Any
 * block that KaTeX refuses to parse fails the CI step.
 *
 * Common failures this catches:
 *   - `\phi^*` being eaten by Markdown's emphasis parser (use `\phi^{\ast}`)
 *   - unbalanced `$$` delimiters
 *   - typos in command names (e.g. `\rangel`)
 *   - missing braces around superscripts / subscripts
 *
 * Usage:  node .github/scripts/check_math.js [file.md ...]
 * With no arguments, scans the whole repo.
 */
const fs = require("fs");
const path = require("path");
const { execSync } = require("child_process");
const katex = require("katex");

function listMarkdownFiles() {
  // Use git to enumerate tracked files so we don't descend into node_modules,
  // .venv, build caches, etc.
  const out = execSync("git ls-files '*.md'", { encoding: "utf8" });
  return out.trim().split("\n").filter(Boolean);
}

function extractMathBlocks(text) {
  /** @type {{kind: "display"|"inline", body: string, line: number}[]} */
  const out = [];
  // Display math first (greedy non-overlapping) so inline matcher skips them.
  const displayRe = /\$\$([\s\S]+?)\$\$/g;
  const redacted = text.replace(displayRe, (match, body, offset) => {
    const line = text.slice(0, offset).split("\n").length;
    out.push({ kind: "display", body, line });
    return " ".repeat(match.length);   // preserve offsets for inline pass
  });
  // Inline math: $...$ on a single line. Skip $$ we already handled.
  const inlineRe = /(?<![\$\\])\$([^\$\n]+?)\$/g;
  let m;
  while ((m = inlineRe.exec(redacted)) !== null) {
    const line = text.slice(0, m.index).split("\n").length;
    out.push({ kind: "inline", body: m[1], line });
  }
  return out;
}

function main() {
  const files = process.argv.slice(2).length
    ? process.argv.slice(2)
    : listMarkdownFiles();

  let total = 0;
  let failed = 0;
  const failures = [];

  for (const file of files) {
    const text = fs.readFileSync(file, "utf8");
    // Reject unbalanced $$ up front — KaTeX won't see a mismatched block.
    const dollarDollars = (text.match(/\$\$/g) || []).length;
    if (dollarDollars % 2 !== 0) {
      failures.push(`${file}: unbalanced $$ (count = ${dollarDollars})`);
      failed++;
      continue;
    }
    for (const { kind, body, line } of extractMathBlocks(text)) {
      total++;
      try {
        katex.renderToString(body, {
          displayMode: kind === "display",
          throwOnError: true,
          strict: "error",
          trust: false,
        });
      } catch (e) {
        failed++;
        failures.push(
          `${file}:${line} [${kind}] ${e.message.split("\n")[0]}\n    body: ${body.replace(/\n/g, " ").slice(0, 160)}`,
        );
      }
    }
  }

  console.log(`Checked ${total} math blocks across ${files.length} files.`);
  if (failed) {
    console.log(`\n${failed} failure(s):`);
    for (const f of failures) console.log("  " + f);
    process.exit(1);
  }
  console.log("All math blocks parse.");
}

main();
