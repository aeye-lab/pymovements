# Copyright (c) 2025 The pymovements Project Authors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
from __future__ import annotations

from warnings import warn


class DeprecatedClassMeta(type):
    def __new__(cls, name, bases, classdict, *args, **kwargs):
        alias = classdict.get('_DeprecatedClassMeta__alias')
        version_deprecated = classdict.get('_DeprecatedClassMeta__version_deprecated')
        version_removed = classdict.get('_DeprecatedClassMeta__version_removed')

        if alias is not None:
            def new(cls, *args, **kwargs):
                alias = getattr(cls, '_DeprecatedClassMeta__alias')
                version_deprecated = getattr(
                    cls, '_DeprecatedClassMeta__version_deprecated',
                )
                version_removed = getattr(
                    cls, '_DeprecatedClassMeta__version_removed',
                )

                if alias is not None:
                    warn(
                        f"{cls.__name__} has been renamed to {alias.__name__} "
                        f"in {version_deprecated} "
                        f"and will be removed in {version_removed}.",
                        DeprecationWarning, stacklevel=2,
                    )

                return alias(*args, **kwargs)

            classdict['__new__'] = new
            classdict['_DeprecatedClassMeta__alias'] = alias
            classdict['_DeprecatedClassMeta__version_deprecated'] = version_deprecated
            classdict['_DeprecatedClassMeta__version_removed'] = version_removed

        fixed_bases = []

        for b in bases:
            alias = getattr(b, '_DeprecatedClassMeta__alias', None)
            version_deprecated = classdict.get('_DeprecatedClassMeta__version_deprecated')
            version_removed = classdict.get('_DeprecatedClassMeta__version_removed')

            if alias is not None:
                warn(
                    f"{cls.__name__} has been renamed to {alias.__name__} "
                    f"in {version_deprecated} "
                    f"and will be removed in {version_removed}.",
                    DeprecationWarning, stacklevel=2,
                )

            # Avoid duplicate base classes.
            b = alias or b
            if b not in fixed_bases:
                fixed_bases.append(b)

        fixed_bases = tuple(fixed_bases)

        return super().__new__(
            cls, name, fixed_bases, classdict, *args, **kwargs,
        )

    def __instancecheck__(cls, instance):
        return any(
            cls.__subclasscheck__(c)
            for c in {type(instance), instance.__class__}
        )

    def __subclasscheck__(cls, subclass):
        if subclass is cls:
            return True
        else:
            return issubclass(
                subclass, getattr(cls, '_DeprecatedClassMeta__alias'),
            )
