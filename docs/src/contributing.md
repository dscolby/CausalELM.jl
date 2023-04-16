# Contributing
CausalELM welcomes and appreciates all contributions. Below are some guidlines to ensure the 
contribution process is as smooth as possible.

## Reporting a Bug
To report a bug, open an issue on the CausalELM.jl GitHub page. Please include all relevant 
infomration, such as what methods were called, the operating system used, the verions of 
CausalELM used, the verion or Julia used, any tracebakcs or error codes, and any other 
relevant information.

## Requesting New Features
Before requesting a new feature, please check the issues page on GitHub to make sure someone
else did not already request the same feature. If this is not the case, then please open an
issue that explains what function or method you would like to be added and how you beleive 
it should behave.

## Contributing Code
Before submitting a pull request, please open an issue explaining what the proposed code is
and why you want to add it. When submitting a pull request, please reference the relevant
issue(s). Please also ensure your code follows the following guidelines.

- All abstract structs, structs, functions, methods, macros, and constants have docstrings 
that follow the same format as the other docstrings. These functions should also be included 
in the relevant section of the API Manual.

- There are no repeated code blocks. If there are repeated codeblocks, then they should be 
in a separate function.

- Methods should generally include types and be type stable.If there is a strong reason to 
deviate from this point, there should be a comment in the code explaining why.

- Minimize use of new constants and macros. If they must be included, the reason for their 
includsion should be obvious or included in the docstring.

- When possible and relevant, code should call the @fastmath and @inbounds macros.

- Use self-explanatory names for variables, methods, structs, constants, and macros.