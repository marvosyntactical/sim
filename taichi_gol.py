import numpy as np

import matplotlib

matplotlib.use('TkAgg', force=True)

import matplotlib.pyplot as plt
import matplotlib.animation as anim
import colorsys as colsys

np.set_printoptions(precision=3)
plt.rcParams['axes.facecolor'] = '#000000' # black background


class Life:

    def __init__(self):

        # settings
        self.dt = 0.01
        self.height = 60
        self.width_to_height_ratio = 16/9.
        self.width = self.width_to_height_ratio*self.height
        self.num_species = 4
        self.particles_per_species = 120
        self.total_particles = self.num_species * self.particles_per_species

        # particle parameters
        self.repelling_force = 4 # force active if r < separation radius
        self.temperature = 0 # controls random fluctuations in particle velocities
        self.friction_coefficient = 20
        self.separation_radius = 25 # mean separation radius
        self.interaction_radius = 25 # mean interaction radius
        self.force_strength = 10 # inter-particle force strength
        self.close_range_factor = 2 # force strength multiplier at r=0
        self.dist_range_factor = 2 # force strength multiplier at r=self.height
        self.deviation = 0.05 # spread in species parameters
        self.seed_range = 0.9 # initial position spread

    def block_matrix(self, p0, p_std, positive=False):
        """
        generates block matrix
        (used for assigning parameters to num_species of particles)
        """
        m = np.random.normal(p0, p_std, (self.num_species, self.num_species))
        m = m.repeat(self.particles_per_species,axis=1).repeat(self.particles_per_species,axis=0)

        if positive:
            return np.abs(m)
        else:
            return m

    def bound(self, x, y):
        """
        periodic boundaries
        """
        x[x>= self.width]  -= 2*self.width
        x[x<=-self.width]  += 2*self.width
        y[y> self.height] -= 2*self.height
        y[y<=-self.height] += 2*self.height

        return x,y

    def attract(self):
        """
        calculate forces between particles
        """

        x = self.x.copy()
        y = self.y.copy()

        # TODO annotate/comment

        dx = x.reshape(self.total_particles, 1) - x # x distance of each particle to all other particles
        dy = y.reshape(self.total_particles, 1) - y
        dx, dy = self.bound(dx, dy) # wrap borders

        r = np.sqrt(dx*dx + dy*dy) # for each pair of particles, their euclidean distance

        #  -------- calculate sum of forces acting on each particle ----------

        # init sum of forces
        f = np.zeros((self.total_particles, self.total_particles))

        i0 = np.where(r < self.separation)
        i1 = np.where((r > self.separation) & (r < (self.separation + self.force_radius/2)))
        i2 = np.where((r > self.separation + self.force_radius/2) & (r < self.separation+self.force_radius))

        f[i0] = self.rep_force[i0]*(self.separation[i0]-r[i0])
        f[i1] = self.forces[i1] * (r[i1]-self.separation[i1])
        f[i2] = -self.forces[i2] * (r[i2]-(self.separation[i2]+self.force_radius[i2]))

        np.fill_diagonal(f, 0) # no self-force

        # increase force at long range
        f *= self.close_range_factor + (self.dist_range_factor-self.close_range_factor) * (r/self.height)

        a = np.arctan2(dy,dx)

        # add contributions from all particles
        fx = np.sum(f * np.cos(a), axis=0)
        fy = np.sum(f * np.sin(a), axis=0)

        return fx, fy

    def friction(self):
        """
        friction and thermal forces

        (adjust friction coeff to reduce wobbling)
        """

        v = np.sqrt(self.vx*self.vx + self.vy * self.vy)
        a = np.arctan2(self.vy, self.vx)

        ffx = self.friction_coefficient * v * np.cos(a) + np.random.normal(0,self.temperature, self.total_particles)
        ffy = self.friction_coefficient * v * np.sin(a) + np.random.normal(0,self.temperature, self.total_particles)

        return ffx, ffy

    def calc_forces(self):
        fx,fy = self.attract()
        ffx,ffy = self.friction()

        fx += ffx
        fy += ffy

        return fx, fy

    def init(self):
        self.particles.set_offsets([])
        self.particles_area.set_offsets([])
        return self.particles, self.particles_area

    def animate(self, i):

        self.bound(self.x, self.y)

        fx, fy = self.calc_forces()

        # UPDATE LOCATIONS
        self.x += self.vx * self.dt
        self.y += self.vy * self.dt
        self.vx -= fx * self.dt
        self.vy -= fy * self.dt

        # update scatter matrices used by anim.FuncAnimation
        xy = np.vstack((self.x,self.y)).T

        self.particles.set_offsets(xy)
        self.particles_area.set_offsets(xy)

        return self.particles, self.particles_area

    def init_species(self):

        # parameter matrices
        self.force_radius = self.block_matrix(self.interaction_radius,self.interaction_radius*self.deviation,True)
        self.separation = self.block_matrix(self.separation_radius, self.separation_radius*self.deviation,True)
        self.forces = self.block_matrix(0,self.force_strength)
        self.rep_force = self.block_matrix(self.repelling_force, self.repelling_force * self.deviation,True)

        # species properties
        colors, sizes, sa = [],[],[]
        num_pps = self.particles_per_species
        for s in range(self.num_species):

            fr = np.abs(self.force_radius[0,s*num_pps])/np.max(self.force_radius)
            fs = np.abs(self.forces[0,s*num_pps])/np.max(np.abs(self.forces))
            sep = np.abs(self.separation[0,s*num_pps])/np.max(self.separation)
            rep = np.abs(self.rep_force[0,s*num_pps])/np.max(self.rep_force)
            fd = np.abs(self.forces[s*num_pps, s*num_pps])/np.max(np.abs(self.forces))

            size = 20 + 50 * fs

            fc = (self.forces[0,s*num_pps]-np.min(self.forces))/(np.max(self.forces)-np.min(self.forces))

            color = colsys.hsv_to_rgb(fc,0.3+0.7*fd,0.5*(1+fs))
            for p in range(num_pps):
                colors.append(color)
                sizes.append(size)
                sa.append(2*fr*15*size)

        self.colors = colors
        self.sizes = sizes
        self.sa = sa # FIXME what is this?? marker radius?


    def init_particles(self):
        # initialization
        self.x = np.random.uniform(-self.height*self.seed_range,self.height*self.seed_range,self.total_particles)
        self.y = np.random.uniform(-self.height*self.seed_range,self.height*self.seed_range,self.total_particles)
        self.vx = np.random.uniform(-1,1,self.total_particles)
        self.vy = np.random.uniform(-1,1,self.total_particles)

    def run(self, maximized=False):

        self.init_species()
        self.init_particles()

        # display parameter settings
        title = ''
        title += f'dt {self.dt:.2f} | '
        title += f'Rep. F {self.repelling_force:.1f} | '
        title += f'Fric. {self.friction_coefficient:.1f} | '
        title += f'Sep. {self.separation_radius:.1f} | '
        title += f'Int. R {self.interaction_radius:.1f} | '
        title += f'F Str. {self.force_strength:.1f} | '
        title += f'Temp. {self.temperature:.1f} | '

        # figure setup
        fig, ax = plt.subplots(figsize=(14,7), facecolor='k')
        plt.title(title)
        # ax.set_aspect('equal')
        plt.xlim(-self.width_to_height_ratio*self.height,self.height*self.width_to_height_ratio)
        plt.ylim(-self.height,self.height)

        self.particles = plt.scatter(self.x, self.y, marker='.', c=self.colors, s=self.sizes)
        self.particles_area = plt.scatter(self.x, self.y, marker='.', c=self.colors, s=self.sa, alpha=0.0, edgecolors=None, linewidths=0)

        animation = anim.FuncAnimation(fig, self.animate, init_func=self.init, interval=0, blit=True)

        plt.axis('off')
        plt.tight_layout(pad=0.3)
        plt.subplots_adjust(hspace=0.01, wspace=0.01)

        if maximized:
            figManager = plt.get_current_fig_manager()
            figManager.window.showMaximized()

        plt.show()

if __name__ == "__main__":
    game = Life()
    game.run()
