import { supabase } from "./supabase";

export interface SavedRoute {
  id: string;
  user_id: string;
  grade: number;
  angle: number;
  image_data_uri: string;
  created_at: string;
  avg_stars: number | null;
  rating_count: number;
  my_stars: number | null;
}

export async function fetchSavedRoutes(userId: string): Promise<SavedRoute[]> {
  if (!supabase) return [];

  const { data: routes, error } = await supabase
    .from("routes")
    .select("id, user_id, grade, angle, image_data_uri, created_at")
    .eq("user_id", userId)
    .order("created_at", { ascending: false });

  if (error) throw error;
  if (!routes || routes.length === 0) return [];

  const routeIds = routes.map((r) => r.id);
  const { data: ratings } = await supabase
    .from("ratings")
    .select("route_id, user_id, stars")
    .in("route_id", routeIds);

  const {
    data: { user },
  } = await supabase.auth.getUser();

  return routes.map((r) => {
    const forRoute = (ratings ?? []).filter((rating) => rating.route_id === r.id);
    const avg = forRoute.length
      ? forRoute.reduce((sum, rating) => sum + rating.stars, 0) / forRoute.length
      : null;
    const mine = forRoute.find((rating) => rating.user_id === user?.id);
    return {
      ...r,
      avg_stars: avg,
      rating_count: forRoute.length,
      my_stars: mine?.stars ?? null,
    };
  });
}

export async function deleteRoute(id: string) {
  if (!supabase) return;
  const { error } = await supabase.from("routes").delete().eq("id", id);
  if (error) throw error;
}

export async function rateRoute(routeId: string, userId: string, stars: number) {
  if (!supabase) return;
  const { error } = await supabase
    .from("ratings")
    .upsert({ route_id: routeId, user_id: userId, stars }, { onConflict: "route_id,user_id" });
  if (error) throw error;
}
