import { useMetrics } from "@/hooks/usePapers";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { MetricsChart } from "@/components/metrics/MetricsChart";
import { Skeleton } from "@/components/ui/skeleton";

export default function Metrics() {
  const { data: metrics, isLoading } = useMetrics();

  if (isLoading) {
    return (
      <div className="space-y-6">
        <Skeleton className="h-[200px] w-full" />
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {Array.from({ length: 3 }).map((_, i) => (
            <Skeleton key={i} className="h-[100px]" />
          ))}
        </div>
      </div>
    );
  }

  if (!metrics) {
    return <div>Failed to load metrics</div>;
  }

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold">Performance Metrics</h1>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <Card>
          <CardHeader>
            <CardTitle>Total Votes</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-3xl font-bold">{metrics.totalVotes}</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Average Relevance</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-3xl font-bold">
              {Math.round(metrics.averageRelevanceScore)}%
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Average Confidence</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-3xl font-bold">
              {Math.round(metrics.averageConfidence)}%
            </p>
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Vote Distribution</CardTitle>
        </CardHeader>
        <CardContent>
          <MetricsChart
            upvotes={metrics.upvotes}
            downvotes={metrics.downvotes}
          />
        </CardContent>
      </Card>
    </div>
  );
}
