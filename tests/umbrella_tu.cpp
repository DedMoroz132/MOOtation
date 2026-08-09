// SPDX-License-Identifier: Apache-2.0
// The umbrella header must be self-contained: including it first, with no
// standard-library headers pulled in beforehand, has to compile. A header that
// only works because the user happened to include <vector> first is broken for
// everyone else.
#include <mootation/mootation.hpp>

#include <iostream>

int main()
{
    std::cout << "MOOtation " << mootation::version() << " — umbrella header OK\n";
    static_assert(MOOTATION_VERSION >= MOOTATION_VERSION_NUM(0, 1, 0),
                  "version macros must be usable in a constant expression");
    return 0;
}
