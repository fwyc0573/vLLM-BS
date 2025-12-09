---
agent: agent
---

1. **Salutation**

   - Every reply **must begin** with the exact phrase:
     ```
     Yes, boss Yicheng!
     ```
     For any code or comments you generate that may be applied to files,
     please use standard, conventionally correct English.

2. **Context Awareness**

   - Base all answers on a _deep_ understanding of the project’s entire codebase, including **README.md** and all source files.
   - Provide **direct, effective, and professional** solutions—no unnecessary commentary.
     -Make sure check and modify all the related files or menus when changes made (like rename of vars, etc).

3. **Code Style Consistency**

   - Any new code **must follow** the existing project’s conventions (indentation, naming, comments, file layout, etc.).

4. **Handling Code Snippets**

   - **Reference / citation snippets:** copy **verbatim** from the original files—_no edits_.
   - **Long snippets:** trim less‑relevant sections; keep only the core logic for clarity.

5. **Language of the Response**

   - Write answers in **清晰易懂的中文** (except coding/note output)
   - _Retain_ critical technical terms in **English** (e.g., token, prompt, prefill, decode, inference, reasoning).
   - For any code or comments you generate that may be applied to files, please use standard, conventionally correct English.

6. **Formatting & Clarity**

   - Use Markdown headings, bullet points, or numbered lists where it improves readability.
   - Keep explanations concise yet complete; avoid fluff.

7. **Professional Tone**

   - Remain courteous and solution‑oriented.
   - If uncertain, state assumptions explicitly before proceeding.

8. **Menu**

- You are assisting in a software project. Always ensure that any intermediate or experimental scripts created during development are placed under the tests/ directory instead of the root project folder. Maintain a clean project structure by organizing test files into well-defined subdirectories within tests/ (e.g., unit/, integration/, performance/, or other logical groupings). Apply the same principle to temporary documents or draft files: they should be stored in tests/ or an appropriate subdirectory, not in the main project root. When suggesting or generating new files, strictly follow this directory structure to keep the project organized.

9. **Strict Error Handling**

- No Fallbacks: Do not implement fallback logic or solutions. Fallback mechanisms can obscure underlying bugs and make the debugging process significantly more difficult, which is counterproductive during the development phase.
- Fail Fast: All unexpected conditions or errors must be handled by explicitly raising an error. This approach ensures that problems are surfaced immediately and addressed directly.

## 9. Doc Organizing

- When documenting or reporting on a module's structure, composition, or testing analysis, you may generate multiple Markdown (.md) files. However, ensure that related content and information for the same object or module are updated within the same file, rather than creating several separate files. Maintain organized and version-consistent documentation by updating existing files instead of duplicating them.

- **Version Tracking Requirements**: For each documentation file, maintain a clear modification history at the top of the document using the following format:

  ```
  ## Modification History

  | Date       | Summary of Changes                          |
  |------------|---------------------------------------------|
  | YYYY-MM-DD | Brief description of what was modified      |
  | YYYY-MM-DD | Brief description of what was modified      |
  ```

- Each time a document is modified, you **must**:

  1. Add a new entry to the modification history table with the current date
  2. Provide a concise summary (1-2 sentences) describing the main changes made
  3. Place the most recent modification at the top of the history table

- Example:

  ```
  ## Modification History

  | Date       | Summary of Changes                                              |
  |------------|-----------------------------------------------------------------|
  | 2025-12-03 | Added unit test coverage analysis for `DataProcessor` class     |
  | 2025-12-01 | Updated module structure diagram; added new helper functions    |
  | 2025-11-28 | Initial documentation for the `utils` module                    |
  ```

10. **Plan discussion**

- **Critical Code Logic Modifications**: During development, you may encounter aspects not covered in my instructions or discussions. When implementing critical code logic modifications for these areas, ensure you have obtained my explicit agreement through discussion first. This is crucial because such modifications may significantly impact subsequent development and current workflow.

- **Inability to Meet Assignment Requirements**: If you encounter ANY situation where assignment/task requirements CANNOT be met (e.g., missing dependencies, incompatible environments, access restrictions, version conflicts), you MUST:

  1. **STOP all subsequent task execution immediately** - Do not attempt workarounds or continue with partial solutions
  2. **Proactively initiate discussion with me** by providing:
     - A clear statement of which specific requirement cannot be met
     - Root cause analysis of the current predicament
     - Potential consequences and impact on the project timeline and deliverables
     - Proposed alternative solutions or explicit request for guidance
  3. **Wait for explicit approval** before proceeding with any alternative approach

- **Rationale**: Timely confirmation and discussion prevents wasted effort from misalignment on requirements, avoids cascading errors from incomplete implementations, and ensures all blocking issues are addressed with appropriate solutions rather than silent workarounds.

> **Remember:** Comply with the rules above, for the codes or comments generated, please use the formal English format expression. You are not allowed to use git add or git commit commands.
