import { ForceGraph2D } from 'react-force-graph';
import { useRef, useCallback, useMemo } from 'react';
import { Paper } from "@db/schema";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";

interface Node {
  id: string;
  title: string;
  category: string;
  relevanceScore: number;
  group: number;
  x?: number;
  y?: number;
}

interface Link {
  source: string;
  target: string;
  value: number;
}

interface GraphData {
  nodes: Node[];
  links: Link[];
}

interface ClusterGraphProps {
  papers: (Paper & {
    relevanceScore?: number;
    similarPapers?: string[];
  })[];
}

export function ClusterGraph({ papers }: ClusterGraphProps) {
  const fgRef = useRef();

  // Prepare graph data
  const graphData = useMemo<GraphData>(() => {
    const nodes: Node[] = papers.map(paper => ({
      id: paper.id.toString(),
      title: paper.title,
      category: paper.primaryCategory,
      relevanceScore: paper.relevanceScore || 0,
      group: getCategoryGroup(paper.primaryCategory)
    }));

    // Create links between papers in the same category and with similar relevance scores
    const links: Link[] = [];
    for (let i = 0; i < papers.length; i++) {
      for (let j = i + 1; j < papers.length; j++) {
        const paper1 = papers[i];
        const paper2 = papers[j];

        // Link papers in the same category
        if (paper1.primaryCategory === paper2.primaryCategory) {
          const relevanceDiff = Math.abs((paper1.relevanceScore || 0) - (paper2.relevanceScore || 0));
          const similarity = 1 - (relevanceDiff / 100);

          if (similarity > 0.7) { // Only link papers with high similarity
            links.push({
              source: paper1.id.toString(),
              target: paper2.id.toString(),
              value: similarity
            });
          }
        }
      }
    }

    return { nodes, links };
  }, [papers]);

  // Custom node painting function
  const paintNode = useCallback(
    (node: any, ctx: CanvasRenderingContext2D) => {
      // Node size based on relevance score
      const size = 4 + (node.relevanceScore / 20);

      ctx.beginPath();
      ctx.arc(0, 0, size, 0, 2 * Math.PI);
      ctx.fillStyle = getCategoryColor(node.category);
      ctx.fill();

      // Add label on hover
      const label = node.title.substring(0, 20) + (node.title.length > 20 ? '...' : '');
      if ((fgRef.current as any)?.centerAt) {
        const graphNode = node as Node & { x?: number; y?: number };
        const isHovered = (fgRef.current as any).centerAt().x === graphNode.x && 
                         (fgRef.current as any).centerAt().y === graphNode.y;
        if (isHovered) {
          ctx.fillStyle = '#fff';
          ctx.font = '4px Sans-Serif';
          ctx.textAlign = 'center';
          ctx.fillText(label, 0, 8);
        }
      }
    },
    []
  );

  return (
    <Card className="w-full">
      <CardHeader>
        <CardTitle>Paper Clustering Visualization</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="h-[600px] w-full">
          <ForceGraph2D
            ref={fgRef as any}
            graphData={graphData}
            nodeLabel="title"
            nodeCanvasObject={paintNode}
            linkWidth={(link: any) => link.value * 2}
            linkColor={() => '#999'}
            backgroundColor="#ffffff"
            nodeRelSize={6}
            linkDirectionalParticles={2}
            linkDirectionalParticleWidth={2}
            d3VelocityDecay={0.3}
          />
        </div>
        <div className="mt-4 flex flex-wrap gap-4">
          {/* Legend */}
          <div className="text-sm">
            <strong>Categories:</strong>
            <div className="flex flex-wrap gap-2 mt-1">
              {CATEGORY_COLORS.map(([category, color]) => (
                <div key={category} className="flex items-center gap-1">
                  <div 
                    className="w-3 h-3 rounded-full" 
                    style={{ backgroundColor: color }}
                  />
                  <span>{category}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

// Category color mapping
const CATEGORY_COLORS = [
  ['cs.LG', '#ff7e79'],
  ['cs.AI', '#4dc9f6'],
  ['cs.CL', '#ffd700'],
  ['cs.CV', '#32CD32'],
  ['cs.NE', '#9966ff'],
] as const;

function getCategoryColor(category: string): string {
  const found = CATEGORY_COLORS.find(([cat]) => category.startsWith(cat));
  return found ? found[1] : '#999';
}

function getCategoryGroup(category: string): number {
  return CATEGORY_COLORS.findIndex(([cat]) => category.startsWith(cat)) + 1;
}