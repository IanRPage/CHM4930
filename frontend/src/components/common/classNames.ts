export function classNames(...names: (string | false | null | undefined)[]) {
  return names.filter(Boolean).join(" ");
}
