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
