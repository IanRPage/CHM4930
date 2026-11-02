import Card from "./Card.tsx";
import "./ComingSoon.css";
import PageHeader from "./PageHeader.tsx";
import Pill from "./Pill.tsx";

export default function ComingSoon({
  title,
  subtitle,
  description,
}: {
  title: string;
  subtitle: string;
  description: string;
}) {
  return (
    <div className="coming-soon">
      <PageHeader title={title} subtitle={subtitle} />
      <Card className="coming-soon-card">
        <Pill tone="neutral">Coming Soon</Pill>
        <p>{description}</p>
      </Card>
    </div>
  );
}
