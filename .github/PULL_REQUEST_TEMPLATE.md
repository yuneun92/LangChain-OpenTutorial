Thank you for contributing to LangChain-OpenTutorial! 

### Pull Request Template    
Follow the template below when opening a PR.

- [ ] **PR title**: "[Team] #-Chapter / #-File Name (without .ipynb extension)"
  - Use "[Team] #-Chapter / #-File Name (without .ipynb extension)" for purely tutorial changes, "infra: ..." for CI changes.
  - Examples
    - Existing Team 3 (Team E): "[E-3] 04-MODEL / 06-HuggingFace Endpoints"
    - New Team 2 (Team N): "[N-2] 07-Text Splitter / 02-RecursiveCharacterTextSplitter"

- [ ] **PR message**: ***Delete this entire checklist*** and Replace the checklist with the following details only if applicable. (즉, 아래 부분은 작성하지 않으셔도 됩니다)
    - **Description:** a description of the change
    - **Issue:** If this PR has a related issue, mention the issue number. (e.g. Fixes #123)
      
Additional guidelines:   
- ❌ Make sure no unnecessary files outside of the tutorial are committed. If unnecessary files (e.g., .bin, .gitignore, poetry.lock, pyproject.toml) are included, the PR must be closed immediately, and the author should reopen it after removal.
- ❌ If the wrong file is uploaded or files are deleted (belongs to someone else), Close the PR immediately and create a new PR.
- 💯 PR Submission & Push Deadline: Every Wednesday 23:59 (The first version must be submitted by Wednesday 23:59.) 
  *If you have personal circumstances, inform the PM in advance. Failure to notify the PM before a delay (by Monday 23:59) may result in a penalty.

### Review Template      
- ⚠️ Author: Before submitting a PR, ensure you have checked all the above points.    
- ✅ Reviewers: Follow this review template when reviewing PRs.

```markdown
🖥️ Review OS: Win/Mac/Linux   
✅ Checklist      
 - [ ] **Template Rule Compliance**: Ensured that the PR follows the template guidelines. (YES/NO)
 - [ ] **Table of Contents Links**: Verified that all links in the Table of Contents work correctly. (YES/NO)
 - [ ] **Image Naming Compliance**: Checked if all included images follow the naming guidelines. (YES/NO)
 - [ ] **Import Statements**: Confirmed that all import statements use the latest version instead of legacy formats. (YES/NO)
 - [ ] **Code Execution**: Verified that all code runs without errors. If warnings occur, mention them in the comments. (YES/NO)     
 - 기타의견: {자유롭게 서술, 한국어 기술 가능}     
```
If no one reviews your PR within a few days, please @-mention one of teddylee777, musangk, BAEM1N
