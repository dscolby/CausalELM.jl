# Contributing
All contributions are welcome. To ensure contributions align with the existing code base and 
are not duplicated, please follow the guidelines below.

## Reporting a Bug
To report a bug, open an issue on the CausalELM.jl GitHub page. Please include all relevant 
information, such as what methods were called, the operating system used, the verion/s of 
CausalELM used, the verion/s of Julia used, any tracebacks or error codes, and any other 
information that would be helpful for debugging. Also be sure to use the bug label.

## Requesting New Features
Before requesting a new feature, please check the issues page on GitHub to make sure someone
else did not already request the same feature. If this is not the case, then please open an
issue that explains what function or method you would like to be added and how you believe 
it should behave. Also be sure to use the enhancement tag.

## Contributing Code
Before submitting a pull request, please open an issue explaining what the proposed code is
and why you want to add it. When submitting a pull request, please reference the relevant
issue/s. Please also ensure your code follows the guidelines below.

*   All abstract structs, structs, functions, methods, macros, and constants have docstrings 
    that follow the same format as the other docstrings. These functions should also be 
    included in the relevant section of the API Manual.

*   There are no repeated code blocks. If there are repeated codeblocks, then they should be 
    in a separate function.

*   Methods should generally include types and be type stable. If there is a strong reason 
    to deviate from this point, there should be a comment in the code explaining why.

*   Minimize use of new constants and macros. If they must be included, the reason for their 
    inclusion should be obvious or included in the docstring.

*   Avoid using global variables and constants.

*   Code should take advantage of Julia's built in macros for performance. Use @inbounds, 
    @view, @fastmath, and @simd when possible.

*   When appending to an array in a loop, preallocate the array and update its values by 
    index.

*   Avoid long functions and decompose them into smaller functions or methods. A general 
    rule is that function definitions should fit within the screen of a laptop.

*   Use self-explanatory names for variables, methods, structs, constants, and macros.