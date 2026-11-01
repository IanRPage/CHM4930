import type { ComponentPropsWithoutRef, ElementType } from "react";
import "./Card.css";
import { classNames } from "./classNames.ts";

type CardProps<T extends ElementType> = {
  as?: T;
} & ComponentPropsWithoutRef<T>;

// White surface with border and shadow. Pages size and pad it with className.
export default function Card<T extends ElementType = "section">({
  as,
  className,
  ...rest
}: CardProps<T>) {
  const Tag: ElementType = as ?? "section";
  return <Tag className={classNames("card", className)} {...rest} />;
}
