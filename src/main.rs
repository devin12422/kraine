use rustc_hash::{FxHashMap, FxHashSet};
use argmin::core::{Error, CostFunction, Gradient, Executor};
use clap::{Args, Parser, ValueEnum};
use std::path::PathBuf;
use argmin::solver::quasinewton::LBFGS;
use nalgebra::Vector3;
use argmin_observer_slog::SlogLogger;
use std::mem;
use std::ops::ControlFlow;
use argmin::core::observers::ObserverMode;
use argmin::solver::linesearch::{MoreThuenteLineSearch,HagerZhangLineSearch};
use pyo3::prelude::*;
use pyo3::types::{IntoPyDict};
use pyo3::ffi::c_str;
use meshopt::{ generate_vertex_remap};
use pyo3::IntoPyObjectExt;
extern crate meshopt;
use rayon::prelude::*;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct DevelopArgs {
    #[arg(value_enum)]
    linesearch: LinesearchType,
    #[arg(value_enum)]
    normalization: Normalization,
    #[arg(short, long, value_name = "FILE")]
    save_as: Option<PathBuf>,
    #[arg(short, long, default_value = "true")]
    use_pairs: bool,
    #[arg(short, long, value_name = "FILE")]
    path: Option<PathBuf>,
    #[arg(short, long,default_value = "0")]
    mesh_index: usize,
    #[arg(short, long,default_value = "10")]
    iters: u64,
    #[arg(long,default_value = "1e-6")]
    c1: f64,
    #[arg(long,default_value = "0.9")]
    c2: f64,
    #[arg(long,default_value = "0.66")]
    gamma: f64,
    #[arg(long,default_value = "0.01")]
    eta: f64,
    #[arg(long,default_value = "1e-10")]
    epsilon: f64,
    #[arg(long,default_value = "0.5")]
    theta: f64,
    #[arg(long,default_value = "1e-10")]
    tol_grad: f64,
    #[arg(long,default_value = "1e-10")]
    tol_cost: f64,
    #[arg(long,default_value = "1e-10")]
    tol_width: f64,
    #[arg(long)]
    with_l1: Option<f64>,
}
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum,Debug)]
enum LinesearchType {
    MoreThuente,
    HagerZhang
}
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum,Debug)]
enum Normalization {
    None,
    All,
    Score,
    Gradient,
}

#[derive(Clone, Debug)]
struct Partition{
    faces: FxHashSet<Face>,
    edges:[[u32;2];2],
    average_normal:Vector3<f64>,
    score:f64,
}

#[derive(PartialEq, Copy, Clone, Hash, Eq, Debug)]
struct Face([u32; 3]);
#[derive( Clone, Debug)]
struct VertexPartition{
    partitions: [Option<Partition>; 2],
    score:f64
}
#[derive(Clone, Debug)]
struct MeshDevelopability {
    faces: Vec<Face>,
    vertex_face_map:FxHashMap<u32,Vec<Face>>,
    normalization: Normalization,
    partition_score:fn(&MeshDevelopability,&mut Partition, &[f64]),
    gradient_fn:fn(&MeshDevelopability, &[f64], &mut[f64],&Partition,&u32),

}
impl MeshDevelopability {
    fn get_vertex_face_map(&self) -> &FxHashMap<u32, Vec<Face>> {
        &self.vertex_face_map
    }

    fn get_faces(&self) -> &Vec<Face> {
        &self.faces
    }
    fn get_edge_adjacent_faces(&self, edge:[u32;2]) ->Vec<&Face>{
        let mut adjacent_faces = Vec::new();
        for face in&self.get_vertex_face_map()[&edge[0]]{
            for rhs in &self.get_vertex_face_map()[&edge[1]]{
                if rhs == face{
                    adjacent_faces.push(face);
                }
            }
        }
        adjacent_faces
    }
    fn get_other_side_of_edge(&self, face:&Face,edge:[u32;2]) ->Option<&Face>{
        let faces = self.get_edge_adjacent_faces(edge);
        if faces.len() <= 1{
            return None
        }
        if faces[0] == face {
            Some(faces[1])
        } else { Some(faces[0]) }
    }
    fn calculate_face_normal(&self,face:&Face,p:&[f64])->Vector3<f64>{
        let i1 = get_vertex_position(p,  face.0[0]);
        let j1 = get_vertex_position(p, face.0[1]);
        let k1 = get_vertex_position(p, face.0[2]);
        // let i_face_index = face.get_face_index_of_index(*i_index).unwrap();
        let eki1 = i1 - k1;
        let eij1 = j1 - i1;
        eij1.cross(&eki1)
    }
    fn calculate_face_index_face_normal(&self,face:&Face,p:&[f64],face_index:usize)->Vector3<f64>{
        let i1 = get_vertex_position(p,  face.0[face_index]);
        let j1 = get_vertex_position(p, face.0[(face_index+1)%3]);
        let k1 = get_vertex_position(p, face.0[(face_index+2)%3]);
        // let i_face_index = face.get_face_index_of_index(*i_index).unwrap();
        let eki1 = i1 - k1;
        let eij1 = j1 - i1;
        eij1.cross(&eki1)
    }
    fn create_vertex_partition_map(&self, p: &[f64]) -> FxHashMap<u32,VertexPartition>{
        let mut vertex_partition_map:FxHashMap<u32,VertexPartition> =FxHashMap::default();
        for partition_face in self.get_faces(){
            'vertex_star_loop : for vertex_star_face_index in 0..3 {
                let vertex_star_index = partition_face.0[vertex_star_face_index];
                let vertex_star_faces = self.get_vertex_face_map().get(&vertex_star_index).unwrap();
                let vertex_star_arity = vertex_star_faces.len();
                if vertex_star_arity <= 3{
                    continue 'vertex_star_loop;
                }
                let vertex_partition = (1..=vertex_star_arity).into_par_iter().map(| big_partition_size|{
                    // The primary partition contains at least one face in a given vertex star, but can contain all of them
                    // For all faces, we iterate over both edges.
                    // There is no mechanism to determine if an edge has been used for partitioning previously.
                    // This doubles the amount of work done per triangle in a given mesh
                    //I believe we can get away with iterating over just 1 edge, but this code lets be lazy AND correct  *hopefully
                    let partition_face_edges = partition_face.get_face_index_face_edges(vertex_star_face_index);
                    partition_face_edges.par_iter().map(|partition_face_edge|{
                        let partition_edge = partition_face.convert_face_edge_to_edge(*partition_face_edge);
                        ([(Some(partition_face.clone()),big_partition_size),(self.get_other_side_of_edge(partition_face,partition_edge).cloned(),vertex_star_arity - big_partition_size)]).par_iter().map(|(face,partition_size)|{
                            if partition_size == &0{
                                ControlFlow::Break(())// We do not iterate over the second partition if it's size is 0 (this may be the second partition)
                            }
                            else {
                                let mut partition = Partition{
                                    faces: FxHashSet::default(),
                                    edges: [partition_edge,partition_edge],
                                    average_normal: Vector3::<f64>::zeros(),
                                    score: 0.0,
                                };
                                //This should change the iteration direction and partition starting face based off of which partition from the set we're currently calculating (i)
                                    // We now have our partition parameters (starting edge, partition size, and iteration direction
                                    // We use a different iterator from the others here, and we have to do the same iteration twice
                                if(0usize..partition_size.clone()).try_fold( face.clone(),|face, _index|{
                                    if let Some(face) = face{
                                        partition.average_normal += self.calculate_face_normal(&face,p);
                                        partition.faces.insert(face.clone());
                                        //Finish up iteration
                                        partition.edges[1] = face.get_face_index_edge_other_edge(face.get_face_index_of_index(vertex_star_index).unwrap(), partition.edges[1]);
                                        ControlFlow::Continue(self.get_other_side_of_edge(&face, partition.edges[1]).cloned())

                                    }else{
                                        ControlFlow::Break(())
                                    }

                                }).is_continue(){
                                    partition.average_normal /= partition.faces.len() as f64; // DO NOT DIVIDE BY VERTEX_ARITY
                                    (self.partition_score)(&self, &mut partition, p);
                                    // // ControlFlow::Continue(partition)

                                    ControlFlow::Continue(partition)
                                }else{
                                    ControlFlow::Break(())
                                }
                            }

                        }).fold(|| { VertexPartition { partitions: [None, None], score: f64::MAX } },|mut vertex_partition, partition|{
                            if let ControlFlow::Continue(partition) = partition{
                                if vertex_partition.partitions[0].is_none(){
                                    vertex_partition.score = partition.score;
                                    vertex_partition.partitions[0] = Some(partition);
                                }else{
                                    if partition.score > vertex_partition.score{
                                        vertex_partition.score = partition.score;
                                    }
                                    vertex_partition.partitions[1] = Some(partition);
                                }
                            }
                            vertex_partition
                        }).reduce(|| { VertexPartition { partitions: [None, None], score: f64::MAX } },|mut vertex_partition_a,mut vertex_partition_b|{
                            if vertex_partition_a.partitions[0].is_none(){
                               vertex_partition_b
                            }else if vertex_partition_b.partitions[0].is_none(){
                                vertex_partition_a
                            }else{
                                if vertex_partition_a.score > vertex_partition_b.score{
                                    vertex_partition_a.partitions[1] = vertex_partition_b.partitions[0].clone();
                                    vertex_partition_a
                                }else{
                                    vertex_partition_b.partitions[1] = vertex_partition_a.partitions[0].clone();
                                    vertex_partition_b
                                }
                            }
                        })
                    }).reduce(|| { VertexPartition { partitions: [None, None], score: f64::MAX } },|mut vertex_partition_a,mut vertex_partition_b|{
                        if vertex_partition_a.partitions[0].is_none(){
                            vertex_partition_b
                        }else if vertex_partition_b.partitions[0].is_none(){
                            vertex_partition_a
                        }else{
                            if vertex_partition_a.score < vertex_partition_b.score{
                                vertex_partition_a.partitions[1] = vertex_partition_b.partitions[0].clone();
                                vertex_partition_a
                            }else{
                                vertex_partition_b.partitions[1] = vertex_partition_a.partitions[0].clone();
                                vertex_partition_b
                            }
                        }
                    })
                }).reduce(|| { VertexPartition { partitions: [None, None], score: f64::MAX } },|mut vertex_partition_a,mut vertex_partition_b|{
                    if vertex_partition_a.partitions[0].is_none(){
                        vertex_partition_b
                    }else if vertex_partition_b.partitions[0].is_none(){
                        vertex_partition_a
                    }else{
                        if vertex_partition_a.score < vertex_partition_b.score{
                            vertex_partition_a.partitions[1] = vertex_partition_b.partitions[0].clone();
                            vertex_partition_a
                        }else{
                            vertex_partition_b.partitions[1] = vertex_partition_a.partitions[0].clone();
                            vertex_partition_b
                        }
                    }
                });
                    // If out of all the partition sizes,
                    // expanding out in either direction, starting from this face,
                    // we have obtained the partition set with the lowest score
                    vertex_partition_map.insert(vertex_star_index, vertex_partition);

            }
        }
        vertex_partition_map
    }
    fn calculate_pair_partition_score(&self, partition: &mut Partition,p: &[f64]) {
        // if score > partition.score {
        //     // The score for a given partition is the maximum of
        //     // the magnitude squared of a face's normal from the average
        //     // for all of it's faces
        //     partition.score = score;
        // }
        // DIVIDE BY PARTITION SIZE
        partition.score = partition.faces.par_iter().map(|face|{
                partition.faces.par_iter().map(|other_face|{
                    if other_face != face {
                            match self.normalization{
                                Normalization::All => {
                                    self.calculate_face_normal(face,p).normalize()-self.calculate_face_normal(other_face, p).normalize()
                                },
                                Normalization::Score => {
                                    self.calculate_face_normal(face,p).normalize()-self.calculate_face_normal(other_face, p).normalize()
                                },
                                _=>{
                                    self.calculate_face_normal(face,p)-self.calculate_face_normal(other_face, p)
                                }
                            }.magnitude_squared()
                    }else{
                        0f64
                    }
                }).reduce(||{0f64}, |a,b|{
                    if a > b {
                        a
                    }else{
                        b
                    }
                })
        }).reduce(||{ 0f64 }, |a, b|{
            if a > b {
                a
            }else{
                b
            }
        }).clone();
    }
    fn calculate_partition_score(&self, partition: &mut Partition,p: &[f64]){
        let pnormal = match self.normalization{
            Normalization::All => {
                partition.average_normal.normalize()
            }
            Normalization::Score => {
                partition.average_normal.normalize()
            }
            _ => {
                partition.average_normal
            }
        };
        partition.score = partition.faces.par_iter().map(|face|{
            match self.normalization{
                Normalization::All => {
                    self.calculate_face_normal(face,p).normalize()-pnormal
                },
                Normalization::Score => {
                    self.calculate_face_normal(face,p).normalize()-pnormal
                },
                _=>{
                    self.calculate_face_normal(face,p)-pnormal
                }
            }.magnitude_squared()
        }).reduce(||{ 0f64 }, |a, b|{
            if a > b {
                a
            }else{
                b
            }
        }).clone();
        // DIVIDE BY PARTITION SIZE
    }
    fn calculate_pair_gradient(&self,p:&[f64],gradients:&mut [f64],partition:&Partition,vertex_star_index:&u32){
        for face1 in partition.faces.iter(){
            let i1_face_index = face1.get_face_index_of_index(vertex_star_index.clone()).unwrap();
            let i1_index = face1.0[i1_face_index];
            let j1_index = face1.0[(i1_face_index+1)%3];
            let k1_index = face1.0[(i1_face_index+2)%3];
            let i1 = get_vertex_position(p, i1_index);
            let j1 = get_vertex_position(p, j1_index);
            let k1 = get_vertex_position(p, k1_index);


            let ejk1 = k1 - j1;
            let eki1 = i1 - k1;
            let eij1 = j1 - i1;
            let mut face1_normal = eij1.cross(&eki1);
            match self.normalization{
                Normalization::All => {
                    face1_normal = face1_normal.normalize();
                }
                Normalization::Gradient => {
                    face1_normal = face1_normal.normalize();
                }
                ,_=>{}
            };
            let area1 = face1_normal.norm();

                for face2 in partition.faces.iter(){
                    if face2 != face1 {
                        let i2_face_index = face2.get_face_index_of_index(vertex_star_index.clone()).unwrap();
                        let i2_index = face2.0[i2_face_index];
                        let j2_index = face2.0[(i2_face_index+1)%3];
                        let k2_index = face2.0[(i2_face_index+2)%3];
                        let i2 = get_vertex_position(p, i2_index);
                        let j2 = get_vertex_position(p, j2_index);
                        let k2 = get_vertex_position(p, k2_index);
                        let ejk2 = k2 - j2;
                        let eki2 = i2 - k2;
                        let eij2 = j2 - i2;
                        let mut face2_normal = eij2.cross(&eki2);
                        // let i_face_index = face.get_face_index_of_index(*i_index).unwrap();
                        match self.normalization{
                            Normalization::All => {
                                face2_normal = face2_normal.normalize();
                            }
                            Normalization::Gradient => {
                                face2_normal = face2_normal.normalize();
                            }
                            ,_=>{}
                        };
                        let lhs = (face1_normal - face2_normal).transpose();

                        // let lhs = face1_normal.normalize() - partition.average_normal;
                        let area2 = face2_normal.norm();
                        let d_ndi1 = ejk1.cross(&face1_normal)* face1_normal.transpose()/ area1;
                        let d_ndj1 = eki1.cross(&face1_normal)* face1_normal.transpose()/ area1;
                        let d_ndk1 = eij1.cross(&face1_normal)* face1_normal.transpose()/ area1;
                        let d_ndi2 = ejk2.cross(&face2_normal)* face2_normal.transpose()/ area2;
                        let d_ndj2 = eki2.cross(&face2_normal)* face2_normal.transpose()/ area2;
                        let d_ndk2 = eij2.cross(&face2_normal)* face2_normal.transpose()/ area2;

                        // let g_ndi = 2f64 * lhs.transpose()* d_ndi;
                        // let g_ndj = 2f64 * lhs.transpose()* d_ndj;
                        // let g_ndk = 2f64 * lhs.transpose()* d_ndk;
                        let gi1 = lhs * d_ndi1;
                        let gj1 = lhs * d_ndj1;
                        let gk1 = lhs * d_ndk1;


                        let gi2 = lhs * d_ndi2;
                        let gj2 = lhs * d_ndj2;
                        let gk2 = lhs * d_ndk2;
                        for li in 0usize..3usize{
                            gradients[i1_index as usize + li] += gi1[li];
                            gradients[k1_index as usize + li] +=gk1[li];
                            gradients[j1_index as usize+ li] += gj1[li];
                            gradients[i2_index as usize + li] -= gi2[li];
                            gradients[k2_index as usize + li] -=gk2[li];
                            gradients[j2_index as usize+ li] -= gj2[li];
                        }
                    }
                }


        }
    }
    fn calculate_gradient(&self,p:&[f64],gradients:&mut [f64],partition:&Partition,vertex_star_index:&u32){
        for face1 in partition.faces.iter(){
            let i1_face_index = face1.get_face_index_of_index(vertex_star_index.clone()).unwrap();
            let i1_index = face1.0[i1_face_index];
            let j1_index = face1.0[(i1_face_index+1)%3];
            let k1_index = face1.0[(i1_face_index+2)%3];
            let i1 = get_vertex_position(p, i1_index);
            let j1 = get_vertex_position(p, j1_index);
            let k1 = get_vertex_position(p, k1_index);


            let ejk1 = k1 - j1;
            let eki1 = i1 - k1;
            let eij1 = j1 - i1;
            let mut face1_normal = eij1.cross(&eki1);
            match self.normalization{
                Normalization::All => {
                    face1_normal = face1_normal.normalize();
                }
                Normalization::Gradient => {
                    face1_normal = face1_normal.normalize();
                }
                ,_=>{}
            };
            let area1 = face1_normal.norm();

            let pnormal = match self.normalization{
                Normalization::All => {
                    partition.average_normal.normalize()
                }
                Normalization::Gradient => {
                    partition.average_normal.normalize()
                }
                _ => {
                    partition.average_normal
                }
            };
            let lhs = face1_normal - pnormal;
            let mut d_ndi = ejk1.cross(&face1_normal)* face1_normal.transpose()/ area1;
            let mut d_ndj = eki1.cross(&face1_normal)* face1_normal.transpose()/ area1;
            let mut d_ndk = eij1.cross(&face1_normal)* face1_normal.transpose()/ area1;
            d_ndi -= d_ndi.unscale(partition.faces.len() as f64);
            d_ndj -= d_ndj.unscale(partition.faces.len() as f64);
            d_ndk -= d_ndk.unscale(partition.faces.len() as f64);
            let g_ndi = lhs.transpose()* d_ndi;
            let g_ndj = lhs.transpose()* d_ndj;
            let g_ndk =lhs.transpose()* d_ndk;
            for li in 0usize..3usize{
                gradients[i1_index as usize + li] += g_ndi.data.0[li][0];
                gradients[k1_index as usize + li] += g_ndk.data.0[li][0];
                gradients[j1_index as usize+ li] += g_ndj.data.0[li][0];
            }
        }
    }
}

impl Face{
    fn get_face_index_edges(&self,index:usize) ->[[u32;2];2]{
        [[self.0[index],self.0[(index+1)%3]],[self.0[index],self.0[(index-1)%3]]]
    }
    fn get_face_index_face_edges(&self,index:usize) ->[[usize;2];2]{
        [[index,(index+1)%3],[index,(index +2 )%3]]
    }
    fn get_face_index_other_face_edge(&self,index:usize,edge:[usize;2]) ->[usize;2]{
        let index_edges = self.get_face_index_face_edges(index);
        if index_edges[0] == edge {
            index_edges[1]
        } else { index_edges[0] }
    }
    fn get_face_index_of_index(&self,vertex_index:u32) ->Option<usize>{
        for index in 0..3{
            if self.0[index] == vertex_index{
                return Some(index);
            }
        }
        println!("Could not find face_index ");
        None
    }
    fn convert_face_edge_to_edge(&self,face_edge:[usize;2])->[u32;2]{
        [self.0[face_edge[0]],self.0[face_edge[1]]]
    }
    fn cmp_edge_and_face_edge(&self,edge:[u32;2],face_edge:[usize;2]) ->bool{

        edge == self.convert_face_edge_to_edge(face_edge)
    }
    fn get_face_index_edge_other_edge(&self,index:usize,edge:[u32;2]) ->[u32;2]{
        let face_edges = self.get_face_index_face_edges(index);
        if self.cmp_edge_and_face_edge(edge, face_edges[0]) {
            self.convert_face_edge_to_edge(face_edges[1])
        } else {
            self.convert_face_edge_to_edge(face_edges[0])
        }
    }
}

#[inline]
fn get_vertex_position(positions:&[f64],vertex_index:u32)->Vector3<f64>{
    Vector3::from([positions[vertex_index as usize],positions[vertex_index as usize +1],positions[vertex_index as usize+2]])
}

impl CostFunction for  MeshDevelopability {
    type Param = Vec<f64>;
    type Output = f64;

    fn cost(&self, p: &Self::Param) -> Result<Self::Output, Error> {

        let partition_map = self.create_vertex_partition_map(p);
        let mut sum = 0f64;
        for vertex_partition in partition_map.values(){
            //println!("Score: {}",vertex_partition.score);
            sum += vertex_partition.score;
        }
        if sum.is_nan() || sum == 0f64{
            sum = f64::MAX;
        }
        //println!("Score: {:?}",sum);
        Ok(sum)
    }
}

impl Gradient for MeshDevelopability {
    type Param = Vec<f64>;
    type Gradient = Vec<f64>;

    fn gradient(&self, p: &Self::Param) -> Result<Self::Gradient, Error> {
        let partition_map = self.create_vertex_partition_map(p);
        let mut gradients = vec![0f64;p.len()];
        for (vertex_star_index,vertex_partition) in partition_map.iter(){
            for partition in vertex_partition.partitions.iter().flatten(){
                (self.gradient_fn)(self,p,gradients.as_mut_slice(),partition,vertex_star_index);

            }
        }
        Ok(gradients)
    }
}
fn main() {
    let args = DevelopArgs::parse();
    println!("Args:{:?}",args);

    let input_mesh_path = args.path.unwrap_or(rfd::FileDialog::new().add_filter("OBJ",&["obj"]).pick_file().unwrap());
    let (models, _) = tobj::load_obj(
        &input_mesh_path,
        &tobj::LoadOptions {
            ignore_lines:true,
            ignore_points:true,
            triangulate: true,
            ..Default::default()
        },
    ).unwrap();
    let loaded_mesh = models[args.mesh_index].clone().mesh;

    let indices:Vec<u32> = vec![0;loaded_mesh.indices.len()];
    let (total_vertices, vertex_remap) = generate_vertex_remap(&loaded_mesh.positions, Some(&loaded_mesh.indices));
    println!("Total vertices: {}",total_vertices);

    unsafe {
        meshopt::ffi::meshopt_remapIndexBuffer(
            indices.as_ptr() as *mut ::std::os::raw::c_uint,
            loaded_mesh.indices.as_ptr() as *mut ::std::os::raw::c_uint,
            loaded_mesh.indices.len(),
            vertex_remap.as_ptr() as *const ::std::os::raw::c_uint,
        );
    }
    println!("Total indices: {}",indices.len());
    let vertices:Vec<[f32;3]> =vec![[0f32,0f32,0f32];total_vertices];
    unsafe {
        meshopt::ffi::meshopt_remapVertexBuffer(
            vertices.as_ptr() as *mut ::std::os::raw::c_void,
            loaded_mesh.positions.as_ptr() as *const ::std::os::raw::c_void,
            total_vertices,
            mem::size_of::<[f32;3]>(),
            vertex_remap.as_ptr() as *const ::std::os::raw::c_uint,
        );
    }
    let mut f64_vertices = Vec::new();
    for vertex in vertices{
        f64_vertices.push([vertex[0] as f64,vertex[1]as f64,vertex[2]as f64]);
    }
    let mut triangles: Vec<[u32;3]> = Vec::new();
    let mut vertex_face_map: FxHashMap<u32,Vec<Face>> = FxHashMap::default();
    let faces = (0..indices.len()/3).map(|face_index|{
        let face = Face([indices[face_index*3],indices[face_index*3+1],indices[face_index*3+2]]);
        triangles.push(face.0);
        for vertex_index in face.0{
            if let Some(mut e) = vertex_face_map.get_mut(&vertex_index) {
               e.push(face);
            } else {
                vertex_face_map.insert(vertex_index, vec![face]);
            }
        }
        face
    }).collect::<Vec<Face>>();
    println!("Total triangles: {}",triangles.len());

    Python::with_gil(|py| {
        let partition_score_fn = if args.use_pairs{
            MeshDevelopability::calculate_pair_partition_score
        }else{
            MeshDevelopability::calculate_partition_score
        };
        let gradient_fn = if args.use_pairs{
            MeshDevelopability::calculate_pair_gradient
        }else{
            MeshDevelopability::calculate_gradient
        };
        let developability_problem = MeshDevelopability{faces,vertex_face_map,partition_score:partition_score_fn,normalization:args.normalization, gradient_fn };
        // let linesearch = argmin::solver::linesearch::HagerZhangLineSearch::new();
            //.with_delta_sigma(0.1,0.2)
            //.unwrap();
         let res_vertices = match args.linesearch{
            LinesearchType::MoreThuente => {
                let linesearch =MoreThuenteLineSearch::new().with_c(args.c1,args.c2).unwrap().with_width_tolerance(args.tol_width).unwrap();
                let tmp_solver = LBFGS::new(linesearch, 8).with_tolerance_cost(args.tol_cost).unwrap().with_tolerance_grad(args.tol_grad).unwrap();
                let solver = if let Some(l1) = args.with_l1{
                    tmp_solver.with_l1_regularization(l1).unwrap()
                }else{
                    tmp_solver
                };

                Executor::new(developability_problem.clone(), solver)
                    .configure(|state| state.param(f64_vertices.into_flattened()).max_iters(args.iters))
                    .add_observer(SlogLogger::term_noblock(), ObserverMode::NewBest)
                    .run().unwrap().state.best_param.unwrap()
            }
            LinesearchType::HagerZhang => {
                let linesearch = HagerZhangLineSearch::new().with_delta_sigma(args.c1,args.c2).unwrap().with_epsilon(args.epsilon).unwrap().with_gamma(args.gamma).unwrap().with_theta(args.theta).unwrap().with_eta(args.eta).unwrap();
                let tmp_solver = LBFGS::new(linesearch, 8).with_tolerance_cost(args.tol_cost).unwrap().with_tolerance_grad(args.tol_grad).unwrap();
                let solver = if let Some(l1) = args.with_l1{
                    tmp_solver.with_l1_regularization(l1).unwrap()
                }else{
                    tmp_solver
                };
                Executor::new(developability_problem.clone(), solver)
                    .configure(|state| state.param(f64_vertices.into_flattened()).max_iters(args.iters))
                    .add_observer(SlogLogger::term_noblock(), ObserverMode::NewBest)
                    .run().unwrap().state.best_param.unwrap()
            }
        };
        let mut grad:Vec<f64> = vec![0f64;total_vertices];
        grad.resize(total_vertices,0f64);
        let mut developed_vertices = Vec::new();
        let partition_map = developability_problem.create_vertex_partition_map(&res_vertices);
        for (index,partition) in partition_map{
            grad[index as usize] = partition.score;
        }
        for index in 0..total_vertices {
            developed_vertices.push([res_vertices[index*3] as f32, res_vertices[index*3+1] as f32,res_vertices[index*3+2] as f32]);
        };

        let ps = py.import("polyscope").unwrap();
        let np = py.import("numpy").unwrap();
        let np_array = np.getattr("array").unwrap();
        let py_vertices = np_array.call1(( developed_vertices.into_bound_py_any(py).unwrap(),)).unwrap();
        let py_indices =np_array.call1(( triangles.into_bound_py_any(py).unwrap(),)).unwrap();
        let py_grad =  np_array.call1((grad.into_bound_py_any(py).unwrap(),)).unwrap();

        let locals = [("ps", py.import("polyscope").unwrap()),].into_py_dict(py).unwrap();
        // ("verts",py_vertices),("faces",py_indices)
        py.eval( c_str!("ps.init()"), None, Some(&locals)).unwrap();

        ps.getattr("register_surface_mesh").unwrap().call1(("my_mesh",py_vertices,py_indices)).unwrap();
        let kwargs = [("defined_on", "vertices"),("cmap","blues")].into_py_dict(py).unwrap();
        py.eval( c_str!("ps.get_surface_mesh('my_mesh')"), None, Some(&locals)).unwrap().getattr("add_scalar_quantity").unwrap().call(("grad",py_grad,),Some(&kwargs)).unwrap();


        py.eval( c_str!("ps.show()"), None, Some(&locals)).unwrap();

    });
    // Print result
}
