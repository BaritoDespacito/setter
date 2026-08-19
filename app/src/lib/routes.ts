import { supabase } from "./supabase";

export interface SavedRoute {
  id: string;
  user_id: string;
  grade: number;
  angle: number;
  image_url: string;
  created_at: string;
  avg_stars: number | null;
  rating_count: number;
  my_stars: number | null;
}

export async function fetchSavedRoutes(userId: string): Promise<SavedRoute[]> {
  if (!supabase) return [];

  const { data: routes, error } = await supabase
    .from("routes")
    .select("id, user_id, grade, angle, image_url, created_at")
    .eq("user_id", userId)
    .order("created_at", { ascending: false });

  if (error) throw error;
  if (!routes || routes.length === 0) return [];

  const routeIds = routes.map((r) => r.id);
  // Independent of each other - run together instead of one-after-another.
  const [{ data: ratings }, { data: userData }] = await Promise.all([
    supabase.from("ratings").select("route_id, user_id, stars").in("route_id", routeIds),
    supabase.auth.getUser(),
  ]);
  const myId = userData.user?.id;

  return routes.map((r) => {
    const forRoute = (ratings ?? []).filter((rating) => rating.route_id === r.id);
    const avg = forRoute.length
      ? forRoute.reduce((sum, rating) => sum + rating.stars, 0) / forRoute.length
      : null;
    const mine = forRoute.find((rating) => rating.user_id === myId);
    return {
      ...r,
      avg_stars: avg,
      rating_count: forRoute.length,
      my_stars: mine?.stars ?? null,
    };
  });
}

// Route images live in Supabase Storage (route-images bucket), not inline in the
// routes table - storing the ~1.2MB base64 PNG directly in a text column made the
// routes list query pull megabytes of text per page load. Object path convention is
// "<user_id>/<route_id>.png", which the bucket's RLS policies key off of.
export async function uploadRouteImage(userId: string, routeId: string, bytes: ArrayBuffer): Promise<string> {
  if (!supabase) throw new Error("Supabase not configured");
  const path = `${userId}/${routeId}.png`;
  const { error } = await supabase.storage
    .from("route-images")
    .upload(path, bytes, { contentType: "image/png", upsert: true });
  if (error) throw error;
  const { data } = supabase.storage.from("route-images").getPublicUrl(path);
  return data.publicUrl;
}

export async function saveRoute(
  userId: string,
  routeId: string,
  grade: number,
  angle: number,
  bytes: ArrayBuffer
): Promise<void> {
  if (!supabase) return;
  const imageUrl = await uploadRouteImage(userId, routeId, bytes);
  const { error } = await supabase
    .from("routes")
    .insert({ id: routeId, user_id: userId, grade, angle, image_url: imageUrl });
  if (error) throw error;
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
