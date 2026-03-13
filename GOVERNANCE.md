# RCMES-MCP Project Governance

This governance model aims to create an open source community that encourages transparency, contributions, and collaboration, while maintaining sound technical and quality standards.

The project follows a liberal contribution model where people and/or organizations who do the most work will have the most influence on project direction. Technical decision making is primarily made through a "[consensus-seeking](https://en.wikipedia.org/wiki/Consensus-seeking_decision-making)" approach.

## Roles

| Role            | Restricted To | Description                                                                 | Read/Clone | Pull Request | Comment | Review | Commit | Decisions |
| --------------- | ------------- | --------------------------------------------------------------------------- | ---------- | ------------ | ------- | ------ | ------ | --------- |
| Contributor     | None          | Anyone providing input: code, issues, documentation, etc.                   | Y          | Y            | Y       |        |        |           |
| Committer       | Contributor   | Contributors granted write access. Responsible for reviewing changes.       | Y          | Y            | Y       | Y      | Y      |           |
| Product Manager | Committer     | Overall manager with final authority when consensus cannot be reached.      | Y          | Y            | Y       | Y      | Y      | Y         |

### Contributor

Contributors include anyone that provides input to the project — code, issues, documentation, designs, or anything else that tangibly improves the project. Start contributing by submitting an [Issue](https://github.com/NASA-JPL/rcmes-mcp/issues) or [Pull Request](https://github.com/NASA-JPL/rcmes-mcp/pulls).

### Committer

Subset of contributors who have been given write access to project repositories. Any contributor who has made a non-trivial contribution should be onboarded as a committer in a timely manner.

#### List of Committers

- Kyongsik Yun ([kyongsik.yun@jpl.nasa.gov](mailto:kyongsik.yun@jpl.nasa.gov)), NASA JPL

### Product Manager

Overall manager of the project with final authority over all key decisions when consensus cannot be reached among committers.

- **Kyongsik Yun**, NASA Jet Propulsion Laboratory

## Acknowledgements

This governance model was adapted from templates provided by [NASA AMMOS SLIM](https://github.com/NASA-AMMOS/slim) and inspired by [node.js](https://github.com/nodejs/node/blob/main/GOVERNANCE.md), [OpenSSL](https://www.openssl.org/policies/omc-bylaws.html), and [OpenMCT](https://github.com/nasa/openmct/blob/master/CONTRIBUTING.md).
