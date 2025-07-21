# Introduction

This is a Cookiecutter template repository for an Apple Metal GPU Project.

# Configurable Variables

The following template variables can be configured

- Project Name
- Project Directory
- Author Name
- Author Email
- License type
- Apple Metal Platform

# Content

The template creates a source directory with separate metal-cpp source folder, cpp source folder, and metal source folder. It also creates an include directory, build directory and bin directory.

The template has a default vector_add kernel that adds two vectors parallely and returns the sum vector. The main cpp file prints the metal device information and runs the test kernel.

The template also comes with a Makefile that can compile the entire repository and run the binary. All files generated during compilation are placed in their appropriate directories.

# Post Completion Hook

As a post completion hook, a git repository is initialized inside the project root folder. A gitignore file is included to ignore the build and bin directories. 

The initial commit adds the default README.md, LICENSE and .gitignore files for tracking with the commit message `"Initial Commit"`
